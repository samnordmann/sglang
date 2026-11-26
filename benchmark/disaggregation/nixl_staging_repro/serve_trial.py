"""Real HTTP model servers: greedy reference, one prefill rank and two decode ranks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import tempfile
import requests
from devices import gpu_ids

p=argparse.ArgumentParser()
p.add_argument("mode",choices=["oracle","pd","slice"])
p.add_argument("out",type=Path)
p.add_argument("--repeats",type=int,default=12)
p.add_argument("--lengths",nargs="+",type=int,default=[64,1024])
p.add_argument("--model",default=os.path.abspath("model"))
p.add_argument("--control-serial",action="store_true")
p.add_argument("--prompt-workload",choices=("random-v1","natural-v1"),
               default=os.environ.get("KV_STUDY_PROMPT_WORKLOAD","random-v1"))
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)

def trial_batches():
    for length in a.lengths:
        for depth in (1,4):
            for rep in range(-2,a.repeats):
                base=1000000+length*10000+depth*1000+(rep+2)*10
                yield [(base+i,length,depth,rep) for i in range(depth)]

natural_inputs={}
if a.prompt_workload == "natural-v1":
    from prompt_workload import TOKENIZER_SHA256, input_hash, load_tokenizer, natural_prompt

    tokenizer=load_tokenizer(a.model)
    for batch in trial_batches():
        for room,length,_,_ in batch:
            ids=natural_prompt(tokenizer,room,length)
            natural_inputs[room]=(ids,input_hash(ids))
    (a.out/"prompt-workload.json").write_text(json.dumps(dict(
        prompt_workload=a.prompt_workload, tokenizer_sha256=TOKENIZER_SHA256,
        input_hash_format="SHA-256 over little-endian uint32 token IDs",
        rooms={room:digest for room,(_,digest) in natural_inputs.items()},
    ), indent=2)+"\n")

def close(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try: child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()

def server(stack,role,port,gpus,trace_dir):
    log=stack.enter_context((a.out/(role+".log")).open("w"))
    cmd=[sys.executable,"-m","sglang.launch_server","--model-path",a.model,
         "--port",str(port),"--tp",str(len(gpus)),"--mem-fraction-static","0.50",
         "--chunked-prefill-size","2048","--max-total-tokens","16384",
         "--disable-radix-cache","--page-size","16","--skip-server-warmup",
         "--cuda-graph-max-bs-decode","4"]
    if role!="oracle":
        cmd+=["--disaggregation-mode",role,"--disaggregation-transfer-backend","nixl",
              "--disaggregation-bootstrap-port","19898","--nccl-port",str(port+500)]
    (a.out/(role+"-cmd.json")).write_text(json.dumps(cmd))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(map(str,gpus)),KV_STUDY_TRACE=trace_dir,
             SGLANG_DISAGGREGATION_NIXL_BACKEND="UCX", SGLANG_DISAGG_STAGING_BUFFER=str(int(a.mode=="pd")),
             SGLANG_ENABLE_NIXL_KV_PACKING="0",
             SGLANG_DISAGG_STAGING_POOL_SIZE_MB="512",TOKENIZERS_PARALLELISM="false")
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    stack.callback(close,proc)
    return proc,port

def wait(servers):
    deadline=time.monotonic()+720
    for child,port in servers:
        while time.monotonic()<deadline:
            if child.poll() is not None: raise RuntimeError(f"server {port} exited {child.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health",timeout=1).ok: break
            except requests.RequestException: pass
            time.sleep(1)
        else: raise TimeoutError(f"server {port} did not start")

def send(port,body):
    r=requests.post(f"http://127.0.0.1:{port}/generate",json=body,timeout=90)
    if not r.ok: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    d=r.json()
    if d.get("meta_info",{}).get("finish_reason",{}).get("type")=="abort": raise RuntimeError(str(d)[:500])
    return d

def run(item, serial=False):
    room,length,depth,rep=item
    wire_room=room+100_000_000 if serial else room
    if a.prompt_workload == "natural-v1":
        ids,digest=natural_inputs[room]
    else:
        rng=random.Random(room)
        ids=[rng.randint(200,10000) for _ in range(length)]
    body=dict(input_ids=ids, sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
    start=time.monotonic_ns()
    if serial:
        body.update(return_logprob=True, top_logprobs_num=2, logprob_start_len=-1)
    if a.mode!="oracle":
        body.update(bootstrap_host="127.0.0.1",bootstrap_port=19898,bootstrap_room=wire_room)
        fut=clients.submit(send,19101,body)
        result=send(19102,body); fut.result()
    else: result=send(19102,body)
    finish=time.monotonic_ns()
    tokens=result.get("output_ids")
    if not tokens or len(tokens)!=2: raise ValueError(str(result)[:600])
    row=dict(room=room,length=length,depth=depth,rep=rep,tokens=tokens,start=start,end=finish)
    if a.prompt_workload == "natural-v1":
        if result["meta_info"].get("prompt_tokens") != length:
            raise ValueError(f"room {room}: server reported an unexpected prompt length")
        row.update(prompt_workload=a.prompt_workload,input_ids_sha256=digest)
    if serial:
        row.update(wire_room=wire_room, output_top_logprobs=result["meta_info"].get("output_top_logprobs"))
    return row

with ExitStack() as cleanup, ThreadPoolExecutor(8) as clients, ThreadPoolExecutor(4) as users:
    trace_dir = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="sglang-kv-events-"))
    def export_traces():
        for event_file in Path(trace_dir).glob("*.jsonl"):
            shutil.copy2(event_file, a.out / event_file.name)
    cleanup.callback(export_traces)
    roles=[("oracle",19102,gpu_ids("decode"))] if a.mode=="oracle" else [("prefill",19101,gpu_ids("prefill")),("decode",19102,gpu_ids("decode"))]
    wait([server(cleanup,*r,trace_dir) for r in roles])
    control_inputs=[]
    with (a.out/"responses.jsonl").open("w") as results:
        for inputs in trial_batches():
            control_inputs.extend(inputs)
            for row in users.map(run,inputs):
                results.write(json.dumps(row)+"\n")
            results.flush()
    if a.control_serial:
        with (a.out/"serial-responses.jsonl").open("w") as results:
            try:
                for item in control_inputs:
                    results.write(json.dumps(run(item,serial=True))+"\n")
                    results.flush()
            except Exception as error:
                (a.out/"serial-control-error.txt").write_text(str(error)+"\n")
