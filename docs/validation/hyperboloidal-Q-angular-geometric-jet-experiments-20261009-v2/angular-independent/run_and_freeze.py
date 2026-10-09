from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
R = P.parents[2]
F = P / 'immutable-independent-Q-angular-gauge-review-20261009'
assert not F.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {p.name: sha(p) for p in [P/'review_angular.py', P/'run_and_freeze.py', P/'REVIEW.md']}
cmd = [sys.executable, str(P/'review_angular.py')]
t = time.monotonic()
run = subprocess.run(cmd, capture_output=True)
seconds = time.monotonic()-t
(P/'review.stdout').write_bytes(run.stdout)
(P/'review.stderr').write_bytes(run.stderr)
receipt = {'command': cmd, 'returncode': run.returncode, 'seconds': seconds,
           'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip(),
           'source_before': before, 'source_after': {n: sha(P/n) for n in before},
           'python_binary_sha256': sha(Path(sys.executable).resolve())}
(P/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
assert run.returncode == 0 and not run.stderr and receipt['source_before'] == receipt['source_after']
F.mkdir()
names = ['review_angular.py', 'run_and_freeze.py', 'REVIEW.md', 'review-results.json',
         'exact-compatible-maps.json', 'receipt.json', 'review.stdout', 'review.stderr']
for name in names:
    shutil.copyfile(P/name, F/name)
entries = [{'path': n, 'sha256': sha(F/n), 'bytes': (F/n).stat().st_size} for n in names]
index = {'kind': 'independent-read-only-angular-gauge-secondjet-review',
         'scientific_index_sha256': '8c569fdf6faf0fa9afff76bebfcd8ea888c5b31734b6fb90b092188f5aab3a65',
         'files': entries, 'scope': 'No kernel rebuild or evolution; explicit exact rank9 compatibility/rank31 basis proof at fixed reference geometry.'}
(F/'index.json').write_text(json.dumps(index, indent=2)+'\n')
for entry in entries:
    assert sha(F/entry['path']) == entry['sha256']
print(json.dumps({'index': str(F/'index.json'), 'sha256': sha(F/'index.json'),
                  'files': len(entries), 'bytes': sum(e['bytes'] for e in entries),
                  'seconds': seconds, 'passed': True}, indent=2))
