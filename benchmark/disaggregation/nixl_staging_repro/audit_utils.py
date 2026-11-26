"""Receipt checks shared by the result validator."""
import json
import math
from plan import LEGACY, batches


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check(ok, text):
    if not ok:
        raise ValueError(text)


def index(items,key):
    indexed={row[key]:row for row in items}
    check(len(items)==len(indexed),f'duplicate {key}')
    return indexed


def quantiles(values):
    def q(p):
        pos=(len(values)-1)*p
        i=math.floor(pos)
        return values[i]+(pos-i)*(values[math.ceil(pos)]-values[i])
    values=sorted(values)
    return dict(n=len(values),p50=q(.5),p95=q(.95))


def workload(plan=LEGACY):
    return {room: (length,depth,rep) for batch in batches(plan) for room,length,depth,rep,_ in batch}


def input_receipts(trial, rows, expected, tokenizer_hash):
    manifest=json.loads((trial/'prompt-workload.json').read_text())
    check(manifest['prompt_workload']=='natural-v1' and manifest['tokenizer_sha256']==tokenizer_hash,
          'natural workload or tokenizer differs')
    check(set(rows)==set(expected),'incomplete natural input cohort')
    result={}
    for room,row in rows.items():
        digest=row.get('input_ids_sha256','')
        check(row.get('prompt_workload')=='natural-v1' and len(digest)==64
              and all(c in '0123456789abcdef' for c in digest)
              and manifest['rooms'].get(str(room))==digest,f'{room}: input fingerprint differs from pre-run manifest')
        result[room]=digest
    return result
