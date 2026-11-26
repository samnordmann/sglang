"""Freeze the shared inter-node workload before the first model invocation."""
import argparse
import hashlib
import json
from pathlib import Path

ORDERS = [["0-baseline", "0-patched"], ["1-patched", "1-baseline"]]
CONTROLS = ["oracle", "oracle-p1d2", "oracle-p1d2-repeat"]
LEGACY = dict(lengths=[64, 1024], depths=[1, 4], repeats=12, warmup=2,
              orders=ORDERS, oracles=CONTROLS, prompt_workload="random-v1")


def load(directory):
    path = directory / "serving-plan.json"
    plan = {**LEGACY, **(json.loads(path.read_text()) if path.is_file() else {})}
    if (plan["depths"] != [1, 4] or plan["warmup"] != 2 or plan["oracles"] != CONTROLS
            or plan["orders"] not in (ORDERS, [list(reversed(order)) for order in ORDERS])
            or type(plan["repeats"]) is not int or not 0 < plan["repeats"] < 295
            or not plan["lengths"] or len(set(plan["lengths"])) != len(plan["lengths"])
            or any(type(n) is not int or n < 16 or n % 16 for n in plan["lengths"])
            or plan["prompt_workload"] not in ("random-v1", "natural-v1")):
        raise ValueError(f"unsupported serving plan: {plan}")
    return plan


def freeze(bundle, output):
    plan = load(bundle)
    if plan["prompt_workload"] == "natural-v1":
        if plan["lengths"] != [64, 2048] or not (bundle/"prompt_workload.py").is_file():
            raise ValueError("the final natural cohort requires lengths 64 and 2048 and its pinned generator")
        plan.update(bundle_plan_sha256=hashlib.sha256((bundle/"serving-plan.json").read_bytes()).hexdigest(),
                    bundle_repeats=plan["repeats"], repeats=24)
    with (output/"serving-plan.json").open("x") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")


def batches(plan):
    for length in plan["lengths"]:
        for depth in plan["depths"]:
            for rep in range(-plan["warmup"], plan["repeats"]):
                base = 1_000_000 + length*10_000 + depth*1000 + (rep+2)*10
                yield [(room, length, depth, rep, base) for room in range(base, base+depth)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "oracle", "runs"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.source, args.output)
        return
    plan = load(args.source)
    if args.action == "runs":
        print(*[name for name in plan["oracles"] if name != "oracle"],
              *(name for order in plan["orders"] for name in order), sep="\n")
    else:
        options = ["--lengths", *plan["lengths"], "--repeats", plan["repeats"]]
        if plan["prompt_workload"] == "natural-v1":
            options.extend(["--prompt-workload", "natural-v1"])
        print(*options, sep="\n")


if __name__ == "__main__":
    main()
