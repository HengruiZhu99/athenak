"""Record the small independent review commands and immutable input pin."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
PY = '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
GATE = P.parent/'conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009'
PIN = 'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = sorted(p for p in P.iterdir() if p.suffix in ('.py', '.md'))
before = {p.name: sha(p) for p in sources}
receipt = {'review_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'gate_index_sha256': PIN, 'production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
           'commands': [], 'source_before': before}
shutil.copyfile(GATE/'index.json', P/'reviewed-gate-index.json')
commands = [([PY, str(P/'math_check.py')], 'symbolic'),
            ([PY, str(P/'pole_check.py'), '--input', str(GATE/'full20.json'), '--output', str(P/'pole-review.json')], 'pole'),
            ([PY, str(P/'verify_gate.py'), '--index', str(GATE/'index.json'), '--sha256', PIN, '--output', str(P/'gate-review.json')], 'gate')]
for command, name in commands:
    start = time.monotonic()
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    for stream in ('stdout', 'stderr'):
        (P/(name+'.'+stream)).write_text(getattr(run, stream))
    receipt['commands'].append({'command': command, 'returncode': run.returncode,
                                'seconds': time.monotonic()-start,
                                'stdout': name+'.stdout', 'stdout_sha256': sha(P/(name+'.stdout')),
                                'stderr': name+'.stderr', 'stderr_sha256': sha(P/(name+'.stderr'))})
    (P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if run.returncode:
        raise SystemExit(run.returncode)
receipt['source_after'] = {p.name: sha(p) for p in sources}
receipt['sources_unchanged'] = receipt['source_after'] == before
assert receipt['sources_unchanged']
receipt['status'] = 'PASS_INDEPENDENT_Q_NULL_LOCAL_REVIEW'
receipt['no_tensor_rerun_no_native_no_propagation'] = True
receipt['scope'] = 'Mathematical/source review and exact reconstruction of pinned existing output; no finite-Fourier transition, Taylor hierarchy, global/native/BH, uniform lapse or energy acceptance.'
(P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'status': receipt['status'], 'commands': len(commands),
                  'seconds': sum(q['seconds'] for q in receipt['commands'])}, indent=2))
