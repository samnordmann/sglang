"""Install two buffered result hooks, preserving all transport and worker sources."""
import hashlib
import json
from pathlib import Path
import sys

tree = Path(sys.argv[1]) / 'python/sglang/srt'
receipt = {}
def change(relative, edits):
    path = tree / relative
    source = path.read_text()
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError((relative, old))
        source = source.replace(old, new)
    path.write_text(source)
    receipt[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

loop = '        for i, (req, next_token_id) in enumerate(\n            zip(batch.reqs, next_token_ids, strict=True)\n        ):\n'
ready = '                req.time_stats.set_prefill_finished_time()\n'
change('disaggregation/prefill.py', [
    (loop, '        import endpoint_trace as et\n        group = next(et.seq)\n' + loop),
    (ready, ready + '                et.save(et.prefill, req.bootstrap_room, next_token_id, group, len(batch.reqs), i)\n')])
done = '        num_generated_tokens = result.get_num_generated_tokens(batch_size)\n        self.metrics_reporter.num_generated_tokens += num_generated_tokens\n        if not batch.spec_algorithm.is_none():\n'
change('managers/scheduler_components/batch_result_processor.py', [
    (done, '        import endpoint_trace as et\n        group = next(et.seq)\n'
     '        for index, (request, token) in enumerate(zip(batch.reqs, next_token_ids, strict=True)):\n'
     '            et.save(et.decode, request.bootstrap_room, token, group, batch_size, index)\n' + done)])
for relative in ('disaggregation/nixl/conn.py', 'disaggregation/common/conn.py'):
    receipt[relative] = hashlib.sha256((tree / relative).read_bytes()).hexdigest()
print(json.dumps(receipt, indent=2))
