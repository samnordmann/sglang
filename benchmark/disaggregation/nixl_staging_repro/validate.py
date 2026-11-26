"""Qualify buffered P1/D2 endpoints, then compare fixed cohorts and burst orders."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import audit_utils as audit
from burst_analysis import analyse as compare_bursts
from clocks import map_event
from plan import batches, load
from transport import attest
from endpoint_trace import limit, record

BASELINE_CONN = '7d7baa3dac2fd8c2bb0fe0fab42078e0a3ab0af4bae37483e2fe09e159e7af7e'
PATCH_CONN = 'a9738e7dd8c01f599bf8258afe410ed17005dc212155b4972eee357d120fc958'
COMMON_CONN = '4da206f7101333a11ace0b32031733247887ba558054861829dbdb93aedcf2a9'
CONTROL = ('oracle-p1d2', 'oracle-p1d2-repeat')


def need(value, what):
    if not value:
        raise ValueError(what)


def events(path, kind):
    raw = path.read_bytes()
    need(len(raw) == record.size * limit, f'wrong allocation {path}')
    by_room, groups = defaultdict(list), defaultdict(dict)
    last = 0
    in_tail = False
    for stamp, room, event, token, size, index, group in record.iter_unpack(raw):
        if not stamp:
            need(not any((room,event,token,size,index,group)), 'incomplete record')
            in_tail = True
            continue
        need(not in_tail and stamp >= last and event == kind and 0 <= index < size
             and index not in groups[group], f'corrupt/out-of-order result {path}')
        last = stamp
        groups[group][index] = room
        by_room[room].append((stamp, token, group, index, size))
    need(by_room and in_tail, f'missing or saturated result buffer {path}')
    return by_room, groups


def result_map(path, role, expected):
    result = {}
    for f in (path / f'trace-{role}').glob('*.resultbin'):
        pid = int(f.stem)
        result[pid] = events(f, 1 if role == 'prefill' else 2)
        need(set(expected) <= result[pid][0].keys(), f'{path}/{role}: absent rooms')
    need(len(result) == (1 if role == 'prefill' else 2), f'{path}/{role}: wrong number of workers')
    return result


def cohort(path, expected, tokenizer, serial=False):
    source = 'serial-responses.jsonl' if serial else 'responses.jsonl'
    rows = audit.index(audit.read(path / source),'room')
    need(set(rows) == set(expected), f'{path}/{source}: unexpected rooms')
    manifest = json.loads((path/'prompt-workload.json').read_text())
    need({int(k) for k in manifest['rooms']} == set(expected), 'wrong natural manifest cohort')
    digests = audit.input_receipts(path, rows, expected, tokenizer)
    for room, row in rows.items():
        need(tuple(row[k] for k in ('length','depth','rep')) == expected[room]
             and len(row['tokens']) == 2 and row['end'] > row['start'], f'{room}: malformed response')
        if serial:
            levels = row['output_top_logprobs']
            need(row['wire_room'] == room + 100_000_000 and len(levels) == 2
                 and all(len(step) == 2 and tok in {r[1] for r in step}
                         and all(isinstance(r[0], (int,float)) and -float('inf') < r[0] <= 0 for r in step)
                         for tok, step in zip(row['tokens'], levels)), f'{room}: invalid serial scores')
    return rows, digests


def analyze_arm(path, expected, node, direct):
    timed, inputs = cohort(path, expected, node['hashes']['tokenizer.json'])
    serial, serial_inputs = cohort(path, expected, node['hashes']['tokenizer.json'], True)
    need(inputs == serial_inputs, 'different serial and timed prompts')
    wires = set(timed) | {r['wire_room'] for r in serial.values()}
    p = result_map(path, 'prefill', wires)
    d = result_map(path, 'decode', wires)
    ppid, (pre, pre_groups) = next(iter(p.items()))
    plog = (path/'prefill.log').read_text(errors='replace')
    dlog = (path/'decode.log').read_text(errors='replace')
    need(re.search(rf'PID{ppid}\] NIXL KVManager initialized with backend: UCX', plog), 'wrong sender PID/UCX')
    for rank in range(2):
        found = re.findall(rf'TP{rank} PID(\d+)\] NIXL KVManager initialized with backend: UCX', dlog)
        need(len(set(found)) == 1 and int(found[0]) in d, 'wrong rank or NIXL backend')
    need(('StagingBuffer allocated:' in plog) != direct and ('StagingBuffer allocated:' in dlog) != direct,
         'stage configuration does not match requested arm')
    cals = audit.index(audit.read(path/'calibration.jsonl'), 'block')
    need(set(cals) == {r['block'] for r in timed.values()}, 'calibration coverage mismatch')
    records = {}
    for is_serial, source in ((False, timed), (True, serial)):
        for room, row in source.items():
            key = row.get('wire_room',room)
            need(len(pre[key]) == 1, 'prefill marker count differs')
            ready, first, group, idx, size = pre[key][0]
            need(row['start'] <= ready < row['end'] and first == row['tokens'][0], 'prefill time/token differs')
            ranks, batches_seen = [], []
            for steps, groups in d.values():
                stamp, tok, gid, pos, width = steps[key][0]
                need(tok == row['tokens'][1], 'first decode token differs from second output token')
                ranks.append(stamp)
                batches_seen.append([width, pos, sorted(groups[gid].items())])
            need(batches_seen[0] == batches_seen[1], 'decode batch members/position differ across ranks')
            if is_serial:
                continue
            expected_block = 1_000_000 + row['length']*10000 + row['depth']*1000 + (row['rep']+2)*10
            need(expected_block == row['block'], f'{room}: incorrect preassigned wave')
            cal = cals[expected_block]
            need(len(cal['before']) == len(cal['after']) == 20, 'incomplete calibration')
            if direct or row['rep'] < 0:
                continue  # direct controls and warmups establish correctness only
            lo, hi = map_event(max(ranks), cal['before'], cal['after'], row['start'], row['end'])
            need(ready <= lo <= hi, f'{room}: decode certainly precedes prefill')
            if row['rep'] >= 0:
                records[room] = dict(room=room, cell=f"L{row['length']}-D{row['depth']}",
                    start_a_ns=row['start'],prefill_a_ns=ready,last_decode_b_ns=max(ranks),
                    http_response_a_ns=row['end'],last_decode_a_bounds_ns=[lo,hi],clock_issue=None,
                    ready_to_decode_us=[(lo-ready)/1000,(hi-ready)/1000],http_us=(row['end']-row['start'])/1000,
                    input_ids_sha256=inputs[room],prefill_batch=[size,idx,sorted(pre_groups[group].items())],
                    first_decode_batch=batches_seen[0])
    sizes = {length*node['config']['num_hidden_layers']*2*(node['config']['num_key_value_heads']//2)
             *node['config']['head_dim']*2 for length,depth,rep in expected.values()}
    # Direct controls use different per-layer writes. Only staged arms are candidates for
    # comparison of UCX protocol tables at the two *expected* bulk payload sizes.
    lanes = [] if direct else attest(plog.splitlines(), sizes, ppid)
    need(direct or (lanes and all(r['direct_rdma'] for r in lanes)),'selected route is not direct RDMA')
    return dict(timed=timed,serial=serial,inputs=inputs,records=records,lanes=lanes)


def summarize(root):
    nodes = [json.loads((root/f'node-{i}.json').read_text()) for i in (0,1)]
    a,b = nodes
    need(a['host'] != b['host'] and a['ip'] != b['ip'] and
         set(a['gpus']).isdisjoint(b['gpus']) and a['revision'] == b['revision'] and
         a['hashes'] == b['hashes'] and a['config'] == b['config'], 'inventory mismatch')
    for i in (0,1):
        for arm, expected_conn in [('baseline',BASELINE_CONN), ('patched',PATCH_CONN)]:
            hooks = json.loads((root/f'hooks-{i}-{arm}.json').read_text())
            need(hooks['disaggregation/nixl/conn.py'] == expected_conn and
                 hooks['disaggregation/common/conn.py'] == COMMON_CONN, 'transport source changed')
    plan = load(root)
    need(plan['lengths'] == [64,2048] and plan['depths'] == [1,4] and plan['repeats'] == 24
         and plan['warmup'] == 2 and plan['prompt_workload'] == 'natural-v1','wrong trial plan')
    arms = [x for order in plan['orders'] for x in order]
    expected = audit.workload(plan)
    mono, hashes = cohort(root/'oracle', expected,a['hashes']['tokenizer.json'])
    runs = {k:analyze_arm(root/k,expected,a,k in CONTROL) for k in (*CONTROL,*arms)}
    need(all(run['inputs'] == hashes for run in runs.values()),'prompt digests differ across trials')
    for earlier,later in zip(arms,arms[1:]):
        need(max(r['end'] for r in runs[earlier]['timed'].values()) <
             min(r['start'] for r in runs[later]['timed'].values()),'arm order contradicted')
    def route(run):
        return sorted((r['src'],r['dst'],r['bytes'],r['description'],r['lane']) for r in run['lanes'])
    need(all(route(runs[k]) == route(runs[arms[0]]) for k in arms),'incomparable selected UCX routes')
    cases = {}
    per_case = {}
    for length in plan['lengths']:
        for depth in plan['depths']:
            cell = f'L{length}-D{depth}'
            rooms = [room for room,param in expected.items() if param[:2] == (length,depth)]
            differences = {phase:{f'{k}/{ref}':[r for r in rooms if data[phase][r]['tokens'] != runs[ref][phase][r]['tokens']]
                                  for k,data in runs.items() for ref in CONTROL if k != ref}
                           for phase in ('timed','serial')}
            valid = not any(ids for phase in differences.values() for ids in phase.values())
            cases[cell] = dict(qualified=valid, comparisons=differences,
                monolithic_differences={k:[r for r in rooms if run['timed'][r]['tokens']!=mono[r]['tokens']]
                                        for k,run in runs.items()},runs={})
            for k in arms:
                rows = [runs[k]['records'][room] for room in rooms if expected[room][2]>=0]
                waves = [[runs[k]['records'][r] for r,_,_,_,_ in wave] for wave in batches(plan)
                         if wave[0][1:3] == (length,depth) and wave[0][3] >= 0]
                departures = [min(r['start_a_ns'] for r in w) for w in waves]
                returns = [max(r['http_response_a_ns'] for r in w) for w in waves]
                spread = [(max(r['start_a_ns'] for r in w)-departures[i])/1000 for i,w in enumerate(waves)]
                pauses = [(departures[i+1]-returns[i])/1000 for i in range(len(waves)-1)]
                need(min(pauses)>=0, 'closed-loop waves overlap')
                cases[cell]['runs'][k] = dict(measurements=rows,
                    within_wave_launch_spread_us=audit.quantiles(spread),between_wave_pause_us=audit.quantiles(pauses), endpoint_lower_us=audit.quantiles([r['ready_to_decode_us'][0] for r in rows]),
                    endpoint_upper_us=audit.quantiles([r['ready_to_decode_us'][1] for r in rows]),
                    http_us=audit.quantiles([r['http_us'] for r in rows]),
                    first_decode_batch_sizes=Counter(r['first_decode_batch'][0] for r in rows))
            per_case[cell]={}
            for order in plan['orders']:
                base=next(k for k in order if k.endswith('baseline'))
                patch=next(k for k in order if k.endswith('patched'))
                pairs=[(runs[base]['records'][r],runs[patch]['records'][r]) for r in rooms if expected[r][2]>=0]
                per_case[cell]['/'.join(order)]={phase:sum(l[phase]!=r[phase] for l,r in pairs)
                    for phase in ('prefill_batch','first_decode_batch')}
    data = dict(qualified=all(case['qualified'] for case in cases.values()),plan=plan,cases=cases,unplanned=[])
    return dict(source=data,bursts=compare_bursts(data),different_batch_compositions_by_room=per_case,
                protocol_tables_for_expected_sizes={k:runs[k]['lanes'] for k in arms})


if __name__=='__main__':
    root=Path(sys.argv[1])
    result=summarize(root)
    (root/'buffered-endpoint-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name, case in result['source']['cases'].items():
        print(name, 'PASS' if case['qualified'] else 'FAIL')
    sys.exit(0 if result['source']['qualified'] else 1)
