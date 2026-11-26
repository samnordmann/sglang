import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from devices import gpu_ids

REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
model, rank, host_a, host_b = sys.argv[1:]
root = Path(model)
hostname = socket.gethostname().split('.')[0]
# Binding the service address checks that this role executes on its designated host.
with socket.socket() as service:
    service.bind((socket.gethostbyname([host_a,host_b][int(rank)]), 0))
ips = [socket.gethostbyname(host) for host in (host_a,host_b)]
if ips[0] == ips[1] or any(ip.startswith('127.') for ip in ips):
    raise ValueError("unusable service addresses")
files = {}
# HF local_dir metadata contains commit, ETag, timestamp. Validate the ETag
# against the bytes, including the Git blob preamble for non-LFS JSON.
for name in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json'):
    path=root/name
    metadata=(root/'.cache/huggingface/download'/(name+'.metadata')).read_text().splitlines()
    revision,etag=metadata[:2]
    if revision!=REVISION or len(etag) not in (40,64):
        raise ValueError(f"unpinned metadata for {name}")
    sha=hashlib.sha256()
    obj=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode()) if len(etag)==40 else sha
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):
            sha.update(block)
            if obj is not sha:
                obj.update(block)
    if obj.hexdigest()!=etag:
        raise ValueError(f"model hash mismatch {name}")
    files[name]=sha.hexdigest()
cfg=json.loads((root/'config.json').read_text())
if cfg.get('model_type')!='qwen3' or cfg.get('torch_dtype',cfg.get('dtype'))!='bfloat16' or cfg['num_key_value_heads']%2:
    raise ValueError("expected Qwen3 BF16, TP2-shardable KV heads")
gpus = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).splitlines()
used = gpu_ids('prefill' if rank=='0' else 'decode')
if any('GB200' not in gpus[idx] for idx in used):
    raise ValueError("unexpected GPU model")
packages = {name: importlib.metadata.version(name) for name in ('torch','sglang-kernel')}
for name in ('nixl', 'nixl-cu13', 'nixl-cu12'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
if not any(key.startswith('nixl') for key in packages):
    raise ValueError("NIXL package version unavailable")
print(json.dumps(dict(host=hostname,ip=ips[int(rank)],revision=REVISION,hashes=files,config=cfg,
                      gpus=[gpus[idx] for idx in used],ucx={k:v for k,v in os.environ.items() if k.startswith('UCX_')},
                      packages=packages)))
