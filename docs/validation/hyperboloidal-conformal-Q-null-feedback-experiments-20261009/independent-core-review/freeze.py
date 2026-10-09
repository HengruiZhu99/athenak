"""Freeze only this completed independent review; never alter reviewed inputs."""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
D = P/'immutable-independent-Q-null-review-20261009'
assert not D.exists()
files = sorted(p for p in P.rglob('*') if p.is_file())
assert json.loads((P/'receipt.json').read_text())['status'] == 'PASS_INDEPENDENT_Q_NULL_LOCAL_REVIEW'
D.mkdir()
entries = {}
for p in files:
    name = str(p.relative_to(P))
    q = D/name
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
    entries[name] = {'sha256': hashlib.sha256(q.read_bytes()).hexdigest(),
                     'bytes': q.stat().st_size}
index = {'scope': 'Independent mathematical/source/hash review only. No tensor rerun, finite-Fourier transition, Taylor hierarchy, propagation/native/BH or uniform lapse acceptance.',
         'reviewed_gate_index_sha256': 'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96',
         'files': entries, 'file_count': len(entries),
         'total_bytes': sum(q['bytes'] for q in entries.values())}
(D/'index.json').write_text(json.dumps(index, indent=2)+'\n')
for name, entry in entries.items():
    assert hashlib.sha256((D/name).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({'index': str(D/'index.json'),
                  'sha256': hashlib.sha256((D/'index.json').read_bytes()).hexdigest(),
                  'files': index['file_count'], 'bytes': index['total_bytes']}, indent=2))
