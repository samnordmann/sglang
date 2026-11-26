"""One fixed-size in-memory record per completed result on the scheduler process."""
import itertools
import mmap
import os
from pathlib import Path
import struct
import time

# KV_STUDY_TRACE is a private per-role directory. The launcher places it on tmpfs.
# The parent copies these mappings after shutdown, even if a scheduler used _exit.
record = struct.Struct("<QQiiiiI")
limit = 4096
prefill = 1
decode = 2
seq = itertools.count(1)
enabled = os.environ.get("KV_MINIMAL_ENDPOINTS") == "1"
storage = None
count = 0


def save(kind, room, token, group, size, index):
    global storage, count
    if not enabled or not isinstance(room, int) or room < 0:
        return
    now = time.monotonic_ns()
    if storage is None:
        folder = Path(os.environ["KV_STUDY_TRACE"])
        if Path("/dev/shm") not in folder.resolve().parents:
            raise RuntimeError("result buffers require a private tmpfs directory")
        with (folder / f"{os.getpid()}.resultbin").open("x+b") as f:
            f.truncate(record.size * limit)
            storage = mmap.mmap(f.fileno(), 0)
        # Fault in the fixed mapping before the first warmup finishes.
        storage[:] = bytes(len(storage))
    if isinstance(token, list):
        if len(token) != 1:
            raise ValueError("only single-token, non-speculative decode is covered")
        token = token[0]
    record.pack_into(storage, count * record.size, now, room, kind, token, size, index, group)
    count += 1
