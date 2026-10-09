"""Freeze the independent fixed-target early-support review once."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
PY = '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
GATE = P.parent/'q-null-early-feedback/immutable-Q-null-early-feedback-local-20261009'
D = P/'immutable-independent-Q-early-support-review-20261009'
assert not D.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
command = [PY, str(P/'review_early.py')]
start = time.monotonic()
run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
for stream in ('stdout', 'stderr'):
    (P/('review.'+stream)).write_text(getattr(run, stream))
receipt = {'command': command, 'returncode': run.returncode,
           'seconds': time.monotonic()-start, 'stdout_sha256': sha(P/'review.stdout'),
           'stderr_sha256': sha(P/'review.stderr'),
           'review_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'no_tensor_native_compiler_or_propagation_executed': True,
           'scope': 'Fixed-target source/principal/pole and saved Fourier contribution review; positive primitive roots retained, no global/native/PDE energy/smooth hierarchy/BH admission.'}
(P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
assert run.returncode == 0 and run.stderr == ''
shutil.copy2(GATE/'index.json', P/'reviewed-index.json')
files = sorted(p for p in P.rglob('*') if p.is_file())
D.mkdir()
entries = {}
for source in files:
    name = str(source.relative_to(P))
    target = D/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    entries[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
index = {'scope': receipt['scope'], 'files': entries, 'file_count': len(entries),
         'bytes': sum(e['bytes'] for e in entries.values()),
         'reviewed_scientific_index_sha256': '902d875e6da80bf3b10e7103aa59cf123c4d8a034862f0a5d9be6f9e220c56c0',
         'helper_sha256': '562d01b4382c5f9c36afc83b29208f5db3d90ca37f12890c92c03b101eb36a69'}
(D/'index.json').write_text(json.dumps(index, indent=2)+'\n')
for name, entry in entries.items():
    assert sha(D/name) == entry['sha256']
print(json.dumps({'index': str(D/'index.json'), 'sha256': sha(D/'index.json'),
                  'files': index['file_count'], 'bytes': index['bytes'],
                  'seconds': receipt['seconds']}, indent=2))
