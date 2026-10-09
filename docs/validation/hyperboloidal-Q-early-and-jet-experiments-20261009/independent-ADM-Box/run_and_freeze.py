"""Run the independent symbolic derivation once and freeze exact evidence."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
root = here.parents[1]
dest = here/'immutable-independent-ADM-Box-null-identity-20261009'
assert not dest.exists()
assert not (here/'radial-identity.json').exists()
sources = [here/'derive_radial.py', here/'first-radial-m1-through-m4.py',
           here/'DERIVATION.md', Path(__file__).resolve()]


def digest(path):
    data = path.read_bytes()
    return {'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)}


before = {str(p.relative_to(here)):digest(p) for p in sources}
command = [sys.executable, str(here/'derive_radial.py')]
started = time.monotonic()
result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
elapsed = time.monotonic()-started
(here/'radial-identity.json').write_bytes(result.stdout)
(here/'run.stderr').write_bytes(result.stderr)
after = {str(p.relative_to(here)):digest(p) for p in sources}
receipt = {'command':command, 'cwd':str(root), 'exit_code':result.returncode,
           'seconds':elapsed, 'stdout':'radial-identity.json', 'stderr':'run.stderr',
           'python':sys.version, 'python_executable':digest(Path(sys.executable)),
           'source_before':before, 'source_after':after, 'sources_unchanged':before==after,
           'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
           'scope':'Symbolic initial Einstein gauge-only ADM/Box identity. No kernel, evolution, general geometric ideal or production adoption.'}
(here/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
assert result.returncode == 0 and not result.stderr and before == after
data = json.loads(result.stdout)
assert data['radial_identity']['arbitrary_F_B_error'] == '0'
assert len(data['rows']) == 18
dest.mkdir()
files = {}
for p in sources+[here/'radial-identity.json',here/'run.stderr',here/'receipt.json']:
    name = str(p.relative_to(here)); payload=p.read_bytes()
    (dest/name).write_bytes(payload)
    assert (dest/name).read_bytes() == payload
    files[name]=digest(p)
(dest/'index.json').write_text(json.dumps({'scope':receipt['scope'],'files':files},indent=2,allow_nan=False)+'\n')
print(json.dumps({'files':len(files),'bytes':sum(s['bytes'] for s in files.values()),
                  'index_sha256':digest(dest/'index.json')['sha256'],
                  'receipt_sha256':digest(dest/'receipt.json')['sha256'], 'seconds':elapsed}))
