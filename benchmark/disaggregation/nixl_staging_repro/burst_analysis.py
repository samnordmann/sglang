"""Offline paired burst analysis of already-qualified two-host serving traces."""
import argparse
import hashlib
import json
from math import comb
from pathlib import Path
from statistics import median

from plan import batches


def endpoints(rows):
    """Return both durations on clock A, in ns. B timestamps select the final decode."""
    first_ready = min(r["prefill_a_ns"] for r in rows)
    last_b = max(r["last_decode_b_ns"] for r in rows)
    # The max of lower bounds is valid for the last event. Among ties for last
    # on B, intersect their upper bounds, since they denote the same B timestamp.
    lower = max(r["last_decode_a_bounds_ns"][0] for r in rows)
    upper = min(r["last_decode_a_bounds_ns"][1] for r in rows if r["last_decode_b_ns"] == last_b)
    if upper < max(lower, first_ready):
        raise ValueError("last decode clock bounds contradict prefill or event ordering")
    decode = (max(lower-first_ready, 0), upper-first_ready)
    http = max(r["http_response_a_ns"] for r in rows) - min(r["start_a_ns"] for r in rows)
    return {"first_prefill_to_last_decode": decode, "first_start_to_last_http": (http, http)}


def rank_interval(n):
    """Largest k with >=95% two-sided binomial coverage for [X_(k), X_(n-k+1)]."""
    best = None
    for k in range(1, (n+1)//2+1):
        coverage = 1 - 2*sum(comb(n, i) for i in range(k))/2**n
        if coverage >= .95:
            best = (k, coverage)
    return best


def us(bounds):
    return [x / 1000 for x in bounds]


def summarize(pairs):
    base, patch = zip(*pairs)
    delta = [(p[0]-b[1], p[1]-b[0]) for b,p in pairs]
    lo, hi = (sorted(values) for values in zip(*delta))
    n = len(delta)
    rank = rank_interval(n)
    def median_bounds(values):
        return us(tuple(median(x) for x in zip(*values)))
    return dict(baseline_median_us=median_bounds(base), patched_median_us=median_bounds(patch),
                paired_median_delta_us=us((median(lo),median(hi))),
                median_delta_order_stat_interval_us=us((lo[rank[0]-1],hi[n-rank[0]])) if rank else None,
                ranks_one_based=[rank[0],n-rank[0]+1] if rank else None,
                nominal_coverage=rank[1] if rank else None,
                definitely_faster=sum(b<0 for _,b in delta), definitely_slower=sum(a>0 for a,_ in delta))


def analyse(source):
    plan = source["plan"]
    orders = plan["orders"]
    names = [name for order in orders for name in order]
    predefined = {f"L{n}-D{d}" for n in plan["lengths"] for d in plan["depths"]}
    if set(source["cases"]) != predefined or set(names) != {"0-baseline","0-patched","1-baseline","1-patched"}:
        raise ValueError("wrong plan cells or run identities")
    results = {}
    for cell, case in source["cases"].items():
        if case["qualified"] is not True:
            results[cell] = {"qualified": False}
            continue
        groups = [[room for room,*_ in batch] for batch in batches(plan)
                  if batch[0][3]>=0 and cell==f"L{batch[0][1]}-D{batch[0][2]}"]
        expected = {room for wave in groups for room in wave}
        if len(groups)!=plan["repeats"] or not groups:
            raise ValueError(f"{cell}: incomplete planned waves")
        waves = {}
        rates = {}
        hashes = {}
        for name in names:
            raw = case["runs"][name]["measurements"]
            rows = {r["room"]:r for r in raw}
            if len(raw)!=len(rows) or set(rows)!=expected:
                raise ValueError(f"{cell}/{name}: duplicate, extra or missing room ID")
            for room,r in rows.items():
                bounds = r["last_decode_a_bounds_ns"]
                if (r["cell"]!=cell or r["clock_issue"] is not None or not bounds or len(bounds)!=2
                        or bounds[0]>bounds[1] or not r["start_a_ns"]<=r["prefill_a_ns"]<=r["http_response_a_ns"]):
                    raise ValueError(f"{cell}/{name}/{room}: invalid clock or stratum in qualified cell")
                if plan.get("prompt_workload")=="natural-v1":
                    digest=r.get("input_ids_sha256")
                    if not digest or hashes.setdefault(room,digest)!=digest:
                        raise ValueError(f"{cell}/{name}/{room}: mismatched input fingerprints")
            previous_end = None
            active_ns = 0
            waves[name] = []
            for group in groups:
                members = [rows[room] for room in group]
                first_start = min(r["start_a_ns"] for r in members)
                last_http = max(r["http_response_a_ns"] for r in members)
                if last_http <= first_start or (previous_end is not None and first_start < previous_end):
                    raise ValueError(f"{cell}/{name}: nonpositive or overlapping closed-loop HTTP waves")
                active_ns += last_http - first_start
                previous_end = last_http
                waves[name].append(endpoints(members))
            rates[name] = dict(measured_requests=len(rows), measured_waves=len(groups),
                               active_http_duration_ns=active_ns,
                               active_http_requests_per_s=len(rows) * 1e9 / active_ns)
        result = {"qualified": True, "paired_wave_count_per_order": len(groups),
                  "runs": rates, "orders": []}
        for order in orders:
            base = next(name for name in order if name.endswith("-baseline"))
            patch = next(name for name in order if name.endswith("-patched"))
            pairs = list(zip(waves[base], waves[patch]))
            metrics = {metric:summarize([(b[metric],p[metric]) for b,p in pairs]) for metric in pairs[0][0]}
            by_wave = []
            for g,(b,p) in zip(groups,pairs):
                by_wave.append(dict(rooms=g,delta_us={m:us((p[m][0]-b[m][1],p[m][1]-b[m][0])) for m in b}))
            result["orders"].append(dict(execution_order=order,metrics=metrics,waves=by_wave))
        results[cell] = result
    return dict(source_global_qualified=source["qualified"], source_unplanned=source.get("unplanned",[]),
                cases=results, interpretation=(
                    "Delta is patched minus baseline for identical room IDs grouped by planned measured wave. "
                    "Ranges propagate the original clock bounds, conditional on the source drift model. "
                    "HTTP events use host A only. Active HTTP requests/s divides the number of measured requests "
                    "by the sum of first-departure to last-return durations within their closed-loop waves, "
                    "excluding warmups, serial controls and all between-wave pauses. This rate is specific "
                    "to the offered concurrency. Order-statistic intervals for the population median assume "
                    "independent paired waves within the given execution order; dependence can reduce coverage. "
                    "The two fixed orders share the same allocation."))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path,help="existing internode-analysis.json")
    p.add_argument("--out",type=Path,help="optional JSON output (default stdout)")
    args=p.parse_args()
    data=args.input.read_bytes()
    result=analyse(json.loads(data))
    result["source_sha256"]=hashlib.sha256(data).hexdigest()
    text=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.out: args.out.write_text(text)
    else: print(text,end="")
