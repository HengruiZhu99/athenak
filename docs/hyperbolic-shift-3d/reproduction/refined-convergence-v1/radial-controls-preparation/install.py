from pathlib import Path
import json,hashlib,tarfile,os,sys,time
root=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/refined-convergence-20261104-v1/stationary-reference-v2');arc=root/'radial-controls-package-v1.tar.gz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(arc)=='e928da71a072799d74a069184b0fa53c0678b99c574d8335075d50f51f4a1b33'
science=json.loads((root/'science.json').read_text());assert science['source_commit']=='b52f09723b9c5a51340401bf91a7bf4426d8f339'
assert sha(root/science['binary'])==science['binary_sha256']=='4220cf70111f829fc8aa3e244f0cd04ad57e2477ca74c294e08c0bb880c7104c'
assert sha(root/'source-manifest.json')==science['source_manifest_sha256']=='19d2445841e7a6e63588a13ab72855fe0da8cd23698c207742f16b16cb3dc608'
assert not (root/'launcher-v7').exists() and not (root/'radial-controls-v1').exists()
with tarfile.open(arc,'r:gz') as t:
 members=t.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
 for m in members:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0] in ('launcher-v7','radial-controls-v1')
 records=json.load(t.extractfile('radial-controls-v1/package-records.json'));assert set(names)==set(records)|{'radial-controls-v1/package-records.json'}
 for m in members:
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(t.extractfile(m).read())
  p.chmod(m.mode)
 for name,r in records.items():
  p=root/name;assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'] and oct(p.stat().st_mode&0o777)==r['mode']
manifest=json.loads((root/'launcher-v7/manifest.json').read_text())
for f,r in manifest.items():assert sha(root/'launcher-v7'/f)==r['sha256']
sys.path.insert(0,str(root/'launcher-v7'));from storage import capacity
beg=time.monotonic();cap=capacity(root,planned=0,reserve=0);elapsed=time.monotonic()-beg
result=dict(status='installed_not_submitted',archive_sha256=sha(arc),files_verified=len(records),launcher_sha256=sha(root/'launcher-v7/manifest.json'),science_sha256=sha(root/'science.json'),storage=cap,storage_seconds=elapsed,qualification='New immutable bounded-control input/launcher namespace; existing launcher-v6/science/binary/job untouched. No build/freeze repeat or physics smoke.')
(root/'radial-controls-v1/install-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
