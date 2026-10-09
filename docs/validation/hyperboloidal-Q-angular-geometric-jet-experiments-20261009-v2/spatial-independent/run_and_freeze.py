from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
F = P/'immutable-independent-Einstein-spatial-pullback-review-20261009'
assert not F.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names = ['review_spatial.py','run_and_freeze.py','REVIEW.md']
before = {name: sha(P/name) for name in names}
cmd = [sys.executable,str(P/'review_spatial.py')]
t = time.monotonic()
run = subprocess.run(cmd,capture_output=True)
seconds = time.monotonic()-t
(P/'review.stdout').write_bytes(run.stdout)
(P/'review.stderr').write_bytes(run.stderr)
receipt = {'command':cmd,'returncode':run.returncode,'seconds':seconds,
           'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'source_before':before,'source_after':{name:sha(P/name)for name in names},
           'python_binary_sha256':sha(Path(sys.executable).resolve()),
           'scope':'Independent read-only source/algebra/100-digit limit review; no actual kernel/native/evolution rerun.'}
(P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert run.returncode==0 and not run.stderr and receipt['source_before']==receipt['source_after']
F.mkdir()
files = names+['review-results.json','receipt.json','review.stdout','review.stderr']
for name in files:
    shutil.copyfile(P/name,F/name)
entries = [{'path':name,'sha256':sha(F/name),'bytes':(F/name).stat().st_size}for name in files]
index = {'kind':'independent-linear-Einstein-spatial-pullback-timejet-review',
         'scientific_index_sha256':'de31bb51e55c159100a58d14acaae0f00231a7d6128bc6c3771d535b75ff5662',
         'files':entries,'scope':receipt['scope']}
(F/'index.json').write_text(json.dumps(index,indent=2)+'\n')
for entry in entries:
    assert sha(F/entry['path'])==entry['sha256']
print(json.dumps({'index':str(F/'index.json'),'sha256':sha(F/'index.json'),
                  'files':len(entries),'bytes':sum(e['bytes']for e in entries),
                  'seconds':seconds,'passed_final_readback':True},indent=2))
