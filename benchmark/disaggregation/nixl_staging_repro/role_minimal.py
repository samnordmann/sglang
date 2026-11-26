"""One independent process per node and trial. A runs the HTTP client."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from devices import gpu_ids
from clocks import respond, sample
from plan import LEGACY, batches, load


def common_server(bundle, out, model, ip, mode):
    # The mono-node script also executes its main loop on import. Reuse only
    # its two pinned lifecycle functions and add the explicit interface bind.
    src = (bundle/'serve_trial.py').read_text()
    functions = [ast.get_source_segment(src, d) for d in ast.parse(src).body
                 if isinstance(d, ast.FunctionDef) and d.name in ('server','close')]
    code = '\n'.join(functions)
    anchor = 'cmd=[sys.executable,"-m","sglang.launch_server",'
    if (len(functions)!=2 or code.count(anchor)!=1 or 'int(a.mode=="pd")' not in code
            or 'SGLANG_ENABLE_NIXL_KV_PACKING="0"' not in code):
        raise ValueError("mono-node launch template changed")
    code = code.replace(anchor,anchor+'"--host",ip,')
    env=dict(os=os,sys=sys,signal=signal,subprocess=subprocess,json=json,
             a=SimpleNamespace(out=out,model=str(model),mode=mode),ip=ip)
    exec(compile(code,str(bundle/'serve_trial.py'),'exec'),env)
    return env['server']


def until(check, child=None, timeout=900):
    limit=time.monotonic()+timeout
    while time.monotonic()<limit:
        if child is not None and child.poll() is not None:
            raise RuntimeError(f"server exited {child.returncode}")
        if check():
            return
        time.sleep(.5)
    raise TimeoutError("peer or server failed to become ready")


def health(ip, port):
    import requests
    try:
        with requests.Session() as s:
            s.trust_env=False
            return s.get(f'http://{ip}:{port}/health', timeout=1).ok
    except requests.RequestException:
        return False


def post(ip, port, body):
    import requests
    with requests.Session() as s:
        s.trust_env=False
        response=s.post(f'http://{ip}:{port}/generate',json=body,timeout=90)
        returned=time.monotonic_ns()   # full response body on A, before future join
        response.raise_for_status()
        result=response.json()
    if result.get('meta_info',{}).get('finish_reason',{}).get('type')=='abort':
        raise ValueError(f"aborted room {body['bootstrap_room']}")
    return result,returned


def traffic(out, ip_a, ip_b, clock, plan=LEGACY, model=None):
    prepared = {}
    if plan["prompt_workload"] == "natural-v1":
        # Loaded from the checksummed single-node bundle on PYTHONPATH.
        from prompt_workload import TOKENIZER_SHA256, input_hash, load_tokenizer, natural_prompt
        tokenizer = load_tokenizer(model)
        for batch in batches(plan):
            for room, length, *_ in batch:
                ids = natural_prompt(tokenizer, room, length)
                prepared[room] = (ids, input_hash(ids))
        with (out/"prompt-workload.json").open("x") as receipt:
            json.dump(dict(prompt_workload="natural-v1", tokenizer_sha256=TOKENIZER_SHA256,
                           rooms={room: value[1] for room, value in prepared.items()}), receipt)
    with ThreadPoolExecutor(4) as clients, ThreadPoolExecutor(4) as users, \
            (out/'responses.jsonl').open('x') as responses,(out/'calibration.jsonl').open('x') as timings:
        def run(item, serial=False):
            room,length,depth,rep,base=item
            wire_room=room+100_000_000 if serial else room
            if prepared:
                ids, digest = prepared[room]
            else:
                rng=random.Random(room)
                ids=[rng.randint(200,10000) for _ in range(length)]
            body=dict(input_ids=ids,
                      sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True),
                      bootstrap_host=ip_a,bootstrap_port=19898,bootstrap_room=wire_room)
            if serial:
                body.update(return_logprob=True,top_logprobs_num=2,logprob_start_len=-1)
            start=time.monotonic_ns()
            prefill=clients.submit(post,ip_a,19101,body)
            result,returned=post(ip_b,19102,body)
            prefill.result()
            row=dict(block=base,room=room,length=length,depth=depth,rep=rep,start=start,
                     end=returned,tokens=result['output_ids'])
            if prepared:
                if result['meta_info'].get('prompt_tokens') != length:
                    raise ValueError(f'room {room}: incorrect prompt token count')
                row.update(prompt_workload="natural-v1", input_ids_sha256=digest)
            if serial:
                row.update(wire_room=wire_room,output_top_logprobs=result['meta_info'].get('output_top_logprobs'))
            return row

        serial_items=[]
        for items in batches(plan):
            serial_items.extend(items)
            before=sample(clock)
            rows=list(users.map(run,items))
            after=sample(clock)
            timings.write(json.dumps(dict(block=items[0][-1],before=before,after=after))+'\n')
            for row in rows:
                responses.write(json.dumps(row)+'\n')
            timings.flush()
            responses.flush()
        with (out/'serial-responses.jsonl').open('x') as serials:
            for item in serial_items:
                serials.write(json.dumps(run(item,serial=True))+'\n')
                serials.flush()



def main():
    rank, bundle, model, root, run, a, b=sys.argv[1:]
    ip_a,ip_b=socket.gethostbyname(a),socket.gethostbyname(b)
    ip=ip_a if rank=='0' else ip_b
    role,port,gpus=('prefill',19101,gpu_ids('prefill')) if rank=='0' else ('decode',19102,gpu_ids('decode'))
    for key in ('NO_PROXY','no_proxy'):
        os.environ[key]=','.join((os.environ.get(key,''),ip_a,ip_b))
    root=Path(root)
    out=root/run
    out.mkdir(exist_ok=True)
    until((root/'oracle.done').exists)
    with tempfile.TemporaryDirectory(prefix='kv-net-events-', dir='/dev/shm') as scratch:
        try:
            with ExitStack() as stack:
                mode='slice' if run.startswith('oracle-p1d2') else 'pd'
                server=common_server(Path(bundle),out,Path(model),ip,mode)
                child,_=server(stack,role,port,gpus,scratch)
                until(lambda:health(ip,port),child)
                if rank=='1':
                    sock=stack.enter_context(socket.socket())
                    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                    sock.bind((ip_b,19899))
                    sock.listen(1)
                    sock.settimeout(900)
                    (out/'decode.ready').touch()
                    conn,peer=sock.accept()
                    if peer[0]!=ip_a:
                        raise ValueError("unexpected client source address")
                    with conn:
                        conn.settimeout(900)
                        respond(conn)
                    until((out/'traffic.done').exists,child,60)
                else:
                    until((out/'decode.ready').exists,child)
                    with socket.create_connection((ip_b,19899),timeout=10,
                                                  source_address=(ip_a,0)) as clock:
                        clock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                        traffic(out,ip_a,ip_b,clock,load(root),model)
                    (out/'traffic.done').touch()
        finally:
            shutil.copytree(scratch,out/f'trace-{role}')
    (out/f'{role}.done').touch()
    peer='decode' if rank=='0' else 'prefill'
    until((out/f'{peer}.done').exists,timeout=90)


if __name__=='__main__':
    main()
