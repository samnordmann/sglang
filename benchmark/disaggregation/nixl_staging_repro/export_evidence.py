"""Export only qualified relative timing statistics and anonymized correctness counts."""

import argparse
import json
import math
from pathlib import Path

from burst_analysis import analyse
from plan import ORDERS


def q(values, p):
    ordered = sorted(values)
    i = (len(ordered) - 1) * p
    a, b = math.floor(i), math.ceil(i)
    return ordered[a] + (i - a) * (ordered[b] - ordered[a])


def request_stats(rows):
    endpoints = [r["ready_to_decode_us"] for r in rows]
    http = [r["http_us"] for r in rows]
    if not all(0 <= lo <= hi and math.isfinite(hi) for lo, hi in endpoints):
        raise ValueError("invalid endpoint durations")
    if not all(t > 0 and math.isfinite(t) for t in http):
        raise ValueError("invalid HTTP durations")
    result = {}
    for name, p in (("p50", .50), ("p95", .95)):
        result[name] = {
            "ready_to_first_decode_us": [q(v, p) for v in zip(*endpoints)],
            "start_to_http_us": q(http, p),
        }
    return result


def mismatch_counts(case):
    if "comparisons" in case:
        comparisons = case["comparisons"]
    else:
        comparisons = {"timed": case["batched_control_differences"],
                       "serial": case["serial_control_differences"]}
    counts = {}
    for phase in ("timed", "serial"):
        pairs = comparisons[phase]
        rooms = {r for ids in pairs.values() for r in ids}
        direct = {r for key, ids in pairs.items() for r in ids
                  if key in ("oracle-p1d2-repeat vs oracle-p1d2",
                             "oracle-p1d2-repeat/oracle-p1d2",
                             "oracle-p1d2/oracle-p1d2-repeat")}
        counts[phase] = {"rooms_with_any_mismatch": len(rooms),
                         "rooms_differing_between_direct_references": len(direct)}
    return counts


def export(raw, pair, baseline_revision, candidate_revision):
    buffered = "bursts" in raw and "source" in raw
    source = raw["source"] if buffered else raw
    plan = source["plan"]
    expected = {"lengths": [64, 2048], "depths": [1, 4], "repeats": 24,
                "warmup": 2, "orders": ORDERS, "prompt_workload": "natural-v1"}
    if any(plan[k] != value for k, value in expected.items()):
        raise ValueError("unexpected comparison design")
    bursts = analyse(source)
    if buffered and bursts != raw["bursts"]:
        raise ValueError("source and archived wave results disagree")
    if pair not in (1, 2):
        raise ValueError("pair must be 1 or 2")
    for sha in (baseline_revision, candidate_revision):
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("expected an exact git revision")
    configuration = {"pair": pair, "instrumentation": "result callbacks only" if buffered else
                     "result callbacks and transfer diagnostics", "baseline": baseline_revision,
                     "candidate": candidate_revision, "model": "Qwen3-0.6B BF16",
                     "hardware": "GB200, one prefill GPU and two remote decode GPUs",
                     "prompt_workload": "natural-v1", "generated_tokens": 2,
                     "warmup_waves_per_cell": 2, "measured_waves_per_cell": 24}
    out = {"configuration": configuration, "all_cells_qualified": bool(source["qualified"]), "cells": {}}
    for length in (64, 2048):
        for depth in (1, 4):
            cell = f"L{length}-D{depth}"
            data = source["cases"][cell]
            paired = bursts["cases"][cell]
            row = {"qualified": paired["qualified"], "checked_requests_per_arm": 26 * depth,
                   "output_checks": mismatch_counts(data)}
            out["cells"][cell] = row
            if not paired["qualified"]:
                continue
            row["orders"] = {}
            for i, order in enumerate(paired["orders"]):
                base = f"{i}-baseline"
                patch = f"{i}-patched"
                label = ("AB", "BA")[i]
                item = {"per_request": {}}
                row["orders"][label] = item
                for run, arm in ((base, "baseline"), (patch, "candidate")):
                    measurements = data["runs"][run]["measurements"]
                    if len(measurements) != 24 * depth:
                        raise ValueError("incomplete timing cohort")
                    item["per_request"][arm] = request_stats(measurements)
                item["active_http_req_per_s"] = {
                    arm: paired["runs"][n]["active_http_requests_per_s"]
                    for arm, n in (("baseline", base), ("candidate", patch))}
                item["paired_waves"] = {}
                for name, metric in (("ready_to_all_first_decode", "first_prefill_to_last_decode"),
                                     ("first_start_to_last_http", "first_start_to_last_http")):
                    summary = order["metrics"][metric]
                    item["paired_waves"][name] = {
                        k: summary[k] for k in ("paired_median_delta_us", "median_delta_order_stat_interval_us",
                                                "ranks_one_based", "nominal_coverage",
                                                "definitely_faster", "definitely_slower")}
                item["per_wave_deltas_us"] = [
                    [*w["delta_us"]["first_prefill_to_last_decode"], w["delta_us"]["first_start_to_last_http"][0]]
                    for w in order["waves"]]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="qualified analysis JSON produced by the harness")
    parser.add_argument("output", type=Path)
    parser.add_argument("--pair", type=int, required=True, choices=(1, 2))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    data = export(json.loads(args.input.read_bytes()), args.pair, args.baseline, args.candidate)
    args.output.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
