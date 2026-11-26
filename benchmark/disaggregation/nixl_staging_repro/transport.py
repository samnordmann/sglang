"""Identify UCX's selected inter-node GPU WRITE protocols for the observed sizes."""
import re

LOG = re.compile(r'\[([^\[\]]+):(\d+):(\d+)\]\s+\|\s*(.*?)\s*\|\s*$')
GPU_WRITE = re.compile(r'remote memory write by ucp_put\S* from (cuda/[^ ]+) to (cuda/[^ ]+)$')
BOUND = re.compile(r'^(\d+)([KMG]?)$',re.I)


def bound(value):
    if value=='inf':
        return float('inf')
    number,suffix=BOUND.fullmatch(value).groups()
    return int(number)*1024**(' KMG'.index(suffix.upper()) if suffix else 0)


def attest(lines, desired, pid):
    configs, active, tables = {}, {}, []
    for line in lines:
        match=LOG.search(line)
        if not match:
            continue
        host,proc,thread,body=match.groups()
        if int(proc)!=pid:
            continue
        origin=(host,proc,thread)
        cols=[col.strip() for col in body.split('|')]
        if len(cols)==1:
            active.pop(origin,None)
            if 'cfg#' in body:
                configs[origin]=body
            elif (gpu:=GPU_WRITE.fullmatch(body)):
                table=dict(config=configs.get(origin,''),source=gpu[1],target=gpu[2],header=body,rows=[])
                active[origin]=table
                tables.append(table)
        elif len(cols)==3 and origin in active and cols[0]!='Range':
            parts=cols[0].split('..')
            if len(parts) not in (1,2):
                continue
            active[origin]['rows'].append((bound(parts[0]),bound(parts[-1]),cols[1],cols[2]))
    rows=[]
    targets=set()
    for table in tables:
        for size in desired:
            for low,high,desc,lane in table['rows']:
                if not low<=size<=high:
                    continue
                if 'inter-node' not in table['config'] or 'cuda_ipc/' in lane or not lane or not desc:
                    raise ValueError(f"inconsistent/incomplete inter-node GPU PUT at {size}: {table['config']}, {desc}, {lane}")
                paths=re.findall(r'([a-z][a-z0-9_]+)/[\w.-]+',lane.lower())
                direct=(bool(paths) and all(p in ('rc_mlx5','dc_mlx5','rc_verbs','dc_verbs','path0','path1') for p in paths)
                        and 'zero-copy' in desc and not re.search('tcp/|cuda_copy|host|stag',(lane+' '+desc).lower())
                        and re.search(r'(?:rc|dc)_(?:mlx5|verbs)/mlx5_\d+:\d+',lane) is not None)
                targets.add(table['target'])
                rows.append(dict(src=table['source'],dst=table['target'],bytes=size,config=table['config'],
                                 header=table['header'],interval=[low,'inf' if high==float('inf') else high],
                                 description=desc,lane=lane,direct_rdma=direct))
    if len(targets)!=2 or any({r['bytes'] for r in rows if r['dst']==t}!=desired for t in targets):
        raise ValueError(f"insufficient GPU protocol tables: {sorted(targets)}, sizes={sorted(desired)}")
    return rows
