"""One-shot compact collection; all accepted source/freeze bytes preserved."""
from pathlib import Path
import hashlib,json,subprocess,math
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/'build-layer-research/continuum/finite-matrix-exact-certificate-20261009'
F=BASE/'immutable-exact-rounded-finite-certificate-20261009'
DEST=ROOT/'docs/validation/hyperboloidal-exact-finite-certificate-experiments-20261009'
FILES={};OMITTED={};PENDING={}
sha=lambda b:hashlib.sha256(b).hexdigest()
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
def queue(source,target,omit=False,spec=None):
 data=source.read_bytes();record={'source':str(source.relative_to(ROOT)),'sha256':sha(data),'bytes':len(data)}
 assert target not in FILES and target not in OMITTED
 if spec:assert record['sha256']==spec['sha256'] and record['bytes']==spec['bytes']
 if omit or len(data)>1048576 or source.suffix in {'.npz','.npy','.o','.a','.bin'}:
  OMITTED[target]=record;return
 if source.suffix=='.json':finite(json.loads(data))
 FILES[target]=record;PENDING[target]=data
assert not DEST.exists()
assert sha((F/'index.json').read_bytes())=='4087edf8fbe887fec4a64dddfe8e98158c2f2c376ca8d5452625888e9cd88208'
index=json.loads((F/'index.json').read_text())
for item in index['files']:queue(F/item['path'],'frozen/'+item['path'],item['role']=='large_payload',item)
queue(F/'index.json','frozen/index.json')
for path in sorted((ROOT/'build-layer-research/finite-certificate-root-review-20261009').iterdir()):
 if path.is_file():queue(path,'root-independent/'+path.name)
for name in ['frozen-readback.stdout','frozen-readback.stderr']:queue(BASE/name,name)
queue(HERE/'README.md','README.md');queue(Path(__file__),'collect_certificate.py')
queue(ROOT/'docs/validation/hyperboloidal-configuration-derivative-experiments-20261009/verify_archive.py','verify_archive.py')
external=json.loads((F/'external-inputs.json').read_text())
records=external if isinstance(external,list) else external['files']
for item in records:
 p=Path(item['path']);data=p.read_bytes();assert len(data)==item['bytes'] and sha(data)==item['sha256']
catalog={'scope':'Exact positive-eigenvalue certificates for rounded J0 finite energy-Galerkin matrices N8/N12/N16; no continuum, unrounded quadrature, subsidiary, native, nonlinear, scri or BH claim.',
 'collection_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':FILES,'omitted_large_payloads':OMITTED,
 'original_frozen_index_sha256':sha((F/'index.json').read_bytes()),'external_input_records_rehashed':records}
finite(catalog);data=(json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode()
DEST.mkdir(parents=True)
for name,value in PENDING.items():
 p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value);assert p.read_bytes()==value
(DEST/'catalog.json').write_bytes(data)
print(json.dumps({'files':len(FILES),'bytes':sum(v['bytes'] for v in FILES.values()),'omitted':len(OMITTED),'external':len(records),'catalog_sha256':sha(data)}))
