"""One-shot compact collection of frozen failed-FD and analytic point evidence."""
from pathlib import Path
import hashlib,json,math,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
F=ROOT/'build-layer-research/continuum/finite-rb-projection-defect/immutable-J0-projection-constraint-defect-20261009'
DEST=ROOT/'docs/validation/hyperboloidal-projection-constraint-defect-experiments-20261009'
FILES={};OMITTED={};PENDING={}
sha=lambda b:hashlib.sha256(b).hexdigest()
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
def queue(source,target,omit=False):
 data=source.read_bytes();assert target not in FILES and target not in OMITTED
 assert not Path(target).is_absolute() and '..' not in Path(target).parts
 spec={'source':str(source.relative_to(ROOT)),'sha256':sha(data),'bytes':len(data)}
 magic=data[:4] in {b'\x7fELF',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xce',b'\xca\xfe\xba\xbe'}
 if omit or magic or data[:8]==b'!<arch>\n' or source.suffix in {'.npz','.npy','.bin','.o','.a','.rst','.raw'} or len(data)>1048576:
  OMITTED[target]=spec;return
 if source.suffix=='.json':finite(json.loads(data))
 FILES[target]=spec;PENDING[target]=data
assert not DEST.exists()
assert sha((F/'index.json').read_bytes())=='970a117d2790f668fc3e086de7477c4ea94b683a8772c27a751f2625a623ef28'
index=json.loads((F/'index.json').read_text());finite(index)
for item in index['files']:
 source=F/item['path'];assert source.stat().st_size==item['bytes'] and sha(source.read_bytes())==item['sha256']
 queue(source,'frozen/'+item['path'],item['role']=='large_payload')
queue(F/'index.json','frozen/index.json')
queue(HERE/'README.md','README.md')
queue(Path(__file__),'collect_projection_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-configuration-derivative-experiments-20261009/verify_archive.py','verify_archive.py')
catalog={'scope':'Same J0/N8/rb=.98 polynomial interpolants: two preserved failed FD attempts and distinct analytic projected-constraint point PASS; general nongauge continuum comparator unresolved. No generator/propagation/stability acceptance.','collection_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':FILES,'omitted_large_payloads':OMITTED,'frozen_index_sha256':sha((F/'index.json').read_bytes()),'frozen_file_count':len(index['files']),'large_data_policy':'Tagged raw call/map/operator/executable payloads, all NPZ/NPY and >1MiB omitted; saved small case records retained for compact arithmetic verification.'}
finite(catalog);data=(json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode()
DEST.mkdir(parents=True)
for target,value in PENDING.items():
 out=DEST/target;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(value);assert out.read_bytes()==value
(DEST/'catalog.json').write_bytes(data)
print(json.dumps({'files':len(FILES),'bytes':sum(v['bytes'] for v in FILES.values()),'omitted_payloads':len(OMITTED),'catalog_sha256':sha(data)}))
