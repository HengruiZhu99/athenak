"""One-shot immutable source/results capsule, including metadata-only payloads."""
from pathlib import Path
import hashlib,json,math,shutil,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=P/'immutable-nonlinear-wave-map-actual-RHS-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
assert not D.exists()
A=P/'attempts/1791564981754626000';r=json.loads((A/'receipt.json').read_text());assert sha(A/'receipt.json')=='518dcec198bf9dde402c70d0b485531162522845388a4e5d3580c39648acdf2a';assert r['passed']
for path,h in r['source_after'].items():assert sha(ROOT/path)==h,path
deps={}
for mode in ['release','debug']:
 b=json.loads((A/('build-'+mode+'.json')).read_text());deps[mode]=b['compiler_dependencies']
 for path,h in deps[mode].items():assert sha(path)==h,path
sources=[s for s in sorted(P.rglob('*'))if s.is_file()and '__pycache__'not in s.parts]
records=[];pending={}
for s in sources:
 assert not s.is_symlink();data=s.read_bytes();name=str(s.relative_to(P));binary=data[:4]in{b'\x7fELF',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xce',b'\xca\xfe\xba\xbe'}or data[:8]==b'!<arch>\n'
 large=binary or len(data)>1048576 or s.suffix in{'.npz','.npy','.jsonl','.o','.a','.bin','.rst'}
 if s.suffix=='.json':finite(json.loads(data))
 records.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'role':'large_payload'if large else'source_or_receipt','origin':str(s)})
 if not large:pending[name]=data
idx={'status':'nonlinear48point actual raw22 source gate PASS; no evolution','launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','accepted_attempt':str(A.relative_to(P)),'accepted_receipt_sha256':sha(A/'receipt.json'),'files':records,'external_source_inputs':r['source_after'],'external_compiler_dependencies':deps,'scope':'finiteOmega source point action only; no principal/evolution/scri/BH claim'}
finite(idx);D.mkdir()
for name,data in pending.items():
 f=D/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data);assert f.read_bytes()==data
(D/'index.json').write_text(json.dumps(idx,indent=2,allow_nan=False)+'\n');print(json.dumps({'index_sha256':sha(D/'index.json'),'small_files':len(pending),'small_bytes':sum(map(len,pending.values())),'metadata_payloads':len(records)-len(pending),'source_inputs':len(r['source_after'])}))
