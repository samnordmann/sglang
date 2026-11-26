"""Deterministic token-exact English archive records for the Qwen3 P/D study."""

import hashlib
from pathlib import Path
import random
import struct


# Qwen/Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca.
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
COLORS = "red blue green gold black white gray yellow amber purple silver pink orange violet cream brown".split()
OBJECTS = "folder ledger notebook binder envelope map chart card album poster journal book atlas stamp ticket label ribbon portrait frame scroll print sketch seal photograph leaflet".split()
TOWNS = "Paris London Rome Berlin Madrid Oslo Boston Denver Bristol York Bath Oxford Leeds Dublin Cork Dover".split()
RECORDS = (
    "The {item} went to {town}.",
    "The clerk sent the {item} to {town}.",
    "The {item} was delivered to {town}.",
    "The office in {town} received the {item}.",
    "A courier brought the {item} to {town}.",
)
NOTES = (
    "", " Each delivery was recorded.", " Staff checked the list at closing time.",
    " The clerk checked the addresses against the register before filing the list.",
    " The archive kept a second copy of the dispatch record for the local offices.",
)
INSTRUCTIONS = (
    "", " Name the town.", " Give its name.", " Give the town name.",
    " Answer with the town name.", " Answer with just the town name.",
    " Give only the name of the town.", " Give only the town name from the records.",
)


def input_hash(ids):
    """SHA-256 of the actual wire token IDs in little-endian uint32 format."""
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def load_tokenizer(model):
    from tokenizers import Tokenizer

    path = Path(model) / "tokenizer.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError(f"{path} does not match the study's pinned Qwen3-0.6B tokenizer")
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def natural_prompt(tokenizer, room, length):
    """Choose complete sentences by token count alone. No generated outputs are used."""
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    rng = random.Random(room)
    items = [f"{color} {obj}" for color in COLORS for obj in OBJECTS]
    rng.shuffle(items)
    focus = items.pop()
    prefix = ("The town archives exchanged materials this week. "
              + rng.choice(RECORDS).format(item=focus, town=rng.choice(TOWNS)))
    question = f"\n\nWhich town received the {focus}?"
    endings = [note + question + instruction for note in NOTES for instruction in INSTRUCTIONS]
    rng.shuffle(endings)
    # All prefixes end in a period. Qwen can merge the period and the newline.
    # Index the incremental lengths with this boundary, then verify the full text.
    by_size = {}
    for ending in endings:
        by_size.setdefault(len(encode("." + ending)) - len(encode(".")), ending)
    size = len(encode(prefix))
    while size + min(by_size) <= length:
        ending = by_size.get(length - size)
        if ending is not None:
            text = prefix + ending
            ids = encode(text)
            if len(ids) == length and tokenizer.decode(ids) == text:
                return ids
        if not items:
            break
        line = " " + rng.choice(RECORDS).format(item=items.pop(), town=rng.choice(TOWNS))
        prefix += line
        size += len(encode(line))
    raise ValueError(f"could not compose an exact {length}-token English prompt for room {room}")


def validate_input_hashes(responses, expected_rooms, workload=None):
    """Require a complete identical input digest for each room across all arms."""
    natural = workload == "natural-v1" or any(
        row.get("prompt_workload") == "natural-v1" for rows in responses.values() for row in rows.values()
    )
    if not natural:
        return False
    if workload not in (None, "natural-v1"):
        raise ValueError(f"natural-v1 responses contradict planned workload {workload}")
    agreed = {}
    for arm, rows in responses.items():
        if rows.keys() != expected_rooms:
            raise ValueError(f"{arm}: incomplete room cohort for the natural-v1 input check")
        for room, row in rows.items():
            digest = row.get("input_ids_sha256", "")
            if (row.get("prompt_workload") != "natural-v1" or not isinstance(digest, str)
                    or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
                raise ValueError(f"{arm} room {room}: missing or invalid natural-v1 input fingerprint")
            if agreed.setdefault(room, digest) != digest:
                raise ValueError(f"{arm} room {room}: input token IDs differ across arms")
    return True
