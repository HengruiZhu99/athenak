"""One-shot compact collection of immutable actual finite-ball matrix evidence."""
from pathlib import Path
import hashlib,json,math,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
F=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009/immutable-total-J-finite-rb-N8-matrix-control-20261009'
DEST=ROOT/'docs/validation/hyperboloidal-finite-rb-energy-Galerkin-experiments-20261009'
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
def tree(folder,target):
 for source in sorted(folder.rglob('*')):
  if source.is_file():queue(source,target+'/'+str(source.relative_to(folder)))
assert not DEST.exists()
review=json.loads((HERE/'independent-draft-review.json').read_text());assert review['passed']
assert sha((HERE/'audit-draft.md').read_bytes())==review['draft_sha256']
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert head=='9bc9fc71b057bc74d1ead0c3b34119390591c1bc'
index=json.loads((F/'index.json').read_text());finite(index)
assert index['production_source_commit']=='27c19d20696ea6dd4704032c51dfd026218f64f2'
for item in index['files']:
 source=F/item['path'];assert source.stat().st_size==item['bytes'] and sha(source.read_bytes())==item['sha256']
 queue(source,'frozen/'+item['path'])
queue(F/'index.json','frozen/index.json')
for item in index['external_large_files']:
 source=Path(item['origin']);assert source.stat().st_size==item['bytes'] and sha(source.read_bytes())==item['sha256']
 if source.suffix in {'.npz','.npy'}:assert item.get('numeric_arrays_finite') is True
for name in ['audit-draft.md','independent-draft-review.json','archive-README.md']:
 queue(HERE/name,('README.md' if name=='archive-README.md' else 'reviews/'+name))
queue(Path(__file__),'collect_matrix_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-configuration-derivative-experiments-20261009/verify_archive.py','verify_archive.py')
catalog={'scope':'Actual full-ball J0/J1/J2 N8 rb=.98 C0 energy-Galerkin source/mass/volume/trace/forcing consistency; no spectrum, propagation, CPBC, exact-scri, continuum or nonlinear stability acceptance.','collection_head':head,'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':FILES,'omitted_large_payloads':OMITTED,'frozen_index_sha256':sha((F/'index.json').read_bytes()),'frozen_file_count':len(index['files']),'frozen_bytes':index['small_file_bytes'],'externally_rehashed_large_payloads':len(index['external_large_files']),'large_data_policy':'All scientific NPZ/NPY and executable bytes omitted regardless of size; raw query/output or >1MiB bytes omitted. Frozen index retains original local path/size/hash/schema.'}
finite(catalog);data=(json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode()
DEST.mkdir(parents=True)
for target,value in PENDING.items():
 out=DEST/target;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(value);assert out.read_bytes()==value
(DEST/'catalog.json').write_bytes(data)
print(json.dumps({'files':len(FILES),'bytes':sum(v['bytes'] for v in FILES.values()),'omitted_frozen_payloads':len(OMITTED),'external_payloads_rehashed':len(index['external_large_files']),'catalog_sha256':sha(data)}))
