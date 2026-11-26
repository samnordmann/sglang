"""Physical CUDA ordinals selected independently for each host."""
import os


def gpu_ids(role):
    key, default, expected = {
        "prefill": ("PREFILL_GPU_IDS", "0", 1),
        "decode": ("DECODE_GPU_IDS", "0,1", 2),
    }[role]
    ids = [int(value) for value in os.getenv(key, default).split(",")]
    if len(set(ids)) != expected or len(ids) != expected or min(ids) < 0:
        raise ValueError(f"{key} must specify {expected} distinct, nonnegative ordinals")
    return ids
