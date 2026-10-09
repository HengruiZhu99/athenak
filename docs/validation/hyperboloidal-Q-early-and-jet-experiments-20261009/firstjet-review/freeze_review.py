"""Run the cheap independent algebra check and freeze its review once."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
PY = '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
G = P.parent/'q-null-jet-followup/immutable-Q-null-firstjet-map-20261009'
D = P/'immutable-independent-Q-firstjet-review-20261009'
assert not D.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
command = [PY, str(P/'review_maps.py')]
start = time.monotonic()
run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
for stream in ('stdout', 'stderr'):
    (P/('review.'+stream)).write_text(getattr(run, stream))
receipt = {'command': command, 'returncode': run.returncode,
           'seconds': time.monotonic()-start,
           'stdout_sha256': sha(P/'review.stdout'),
           'stderr_sha256': sha(P/'review.stderr'),
           'review_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'no_tensor_native_compiler_or_propagation_executed': True,
           'scope': 'Linear analytic-reference first-jet map and physical ADM/null condition reconstruction; no nonlinear or exact Einstein ideal invariance conclusion.'}
(P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
assert run.returncode == 0 and run.stderr == ''
shutil.copy2(G/'index.json', P/'reviewed-index.json')
files = sorted(p for p in P.rglob('*') if p.is_file())
D.mkdir()
entries = {}
for source in files:
    name = str(source.relative_to(P))
    target = D/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    entries[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
index = {'scope': receipt['scope'], 'files': entries,
         'file_count': len(entries), 'bytes': sum(e['bytes'] for e in entries.values()),
         'reviewed_scientific_index_sha256': '5490057dfc04e060bca65ec7ee1a3bb363a34cab0fe5c890f23d477a7bda7ec1'}
(D/'index.json').write_text(json.dumps(index, indent=2)+'\n')
for name, entry in entries.items():
    assert sha(D/name) == entry['sha256']
print(json.dumps({'index': str(D/'index.json'), 'sha256': sha(D/'index.json'),
                  'files': index['file_count'], 'bytes': index['bytes'],
                  'seconds': receipt['seconds']}, indent=2))
