"""Freeze the completed independent review without touching its reviewed inputs."""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
D = P / 'immutable-independent-flat-penrose-review-20261009'
assert not D.exists()
files = [p for p in sorted(P.rglob('*')) if p.is_file()]
D.mkdir()
entries = {}
for p in files:
    name = str(p.relative_to(P))
    q = D / name
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
    entries[name] = {'sha256': hashlib.sha256(q.read_bytes()).hexdigest(),
                     'bytes': q.stat().st_size}
index = {'scope': 'Read-only symbolic geometry/endpoint/prior-family review; no native or stability gate',
         'files': entries, 'count': len(entries),
         'bytes': sum(q['bytes'] for q in entries.values())}
(D / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
for name, q in entries.items():
    assert hashlib.sha256((D / name).read_bytes()).hexdigest() == q['sha256']
print(D / 'index.json')
print(hashlib.sha256((D / 'index.json').read_bytes()).hexdigest())
print(index['count'], index['bytes'])
