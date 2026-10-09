"""One-shot compact collection of frozen continuum rates and actual API builds."""
from pathlib import Path
import hashlib,json,math,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
P=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
F=ROOT/'build-layer-research/continuum/finite-rb-constraint-rate-oracle/immutable-finite-rb-C0-constraint-rates-20261009'
ORACLE=F.parent
DEST=ROOT/'docs/validation/hyperboloidal-continuum-constraint-rate-experiments-20261009'
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
assert (HERE/'independent-draft-review.json').is_file()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert head=='5ef1c97d10f17a8af0890e5cb39ac463b25ee084'
assert sha((F/'index.json').read_bytes())=='3d4c613a814a8a3325a7f980c2e20dcabf3ea08ddbcb10d42026ddca732d4e2f'
index=json.loads((F/'index.json').read_text())
for item in index['files']:
 source=F/item['path'];assert source.stat().st_size==item['bytes'] and sha(source.read_bytes())==item['sha256']
 queue(source,'frozen/'+item['path'],item['role']=='large_payload')
queue(F/'index.json','frozen/index.json')
deps={}
for mode,attempt in [('release','release-005'),('debug','debug-003')]:
 b=P/'build-attempts'/attempt;r=json.loads((b/'receipt.json').read_text())
 assert r['exit_code']==0 and r['sources_before']==r['sources_after']
 assert sha((b/('radial-bridge-'+mode)).read_bytes())==r['executable_sha256']
 tree(b,'build-attempts/'+attempt)
 for name,digest in (r['compiler_dependency_hashes']|r['link_archive_hashes']).items():
  source=Path(name)
  if source.parent==P and (b/source.name).is_file():source=b/source.name
  assert sha(source.read_bytes())==digest
  if name in deps:assert deps[name][0]==digest
  deps[name]=(digest,source)
for name,(digest,source) in deps.items():
 original=Path(name)
 if original.is_relative_to(ROOT):
  relative=original.relative_to(ROOT)
  if relative.parts[0]!='kokkos' and source.suffix not in {'.a','.o'}:
   queue(source,'as-built-inputs/'+str(relative))
for name in ['frozen-readback-receipt.json','frozen-readback.stdout.json','frozen-readback.stderr']:
 queue(ORACLE/name,'reviews/original-'+name)
for name in ['audit-draft.md','archive-README.md','source-inventory.json','summary.json','independent-draft-review.json','independent_review.py','independent-review.stdout','independent-review.stderr','review_build_provenance.py','root-provenance-review.json','root-saved-data-readback.stdout.json','root-saved-data-readback.stderr']:
 queue(HERE/name,'reviews/'+name)
queue(HERE/'archive-README-final.md','README.md')
queue(Path(__file__),'collect_constraint_rate_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-configuration-derivative-experiments-20261009/verify_archive.py','verify_archive.py')
catalog={'scope':'C0 stationary-Minkowski-reference physical8 continuum constraint-rate core/gauge/shell checks; no radial projection,SAT,propagation,native pulse,BH or exact-scri admission.','collection_head':head,'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','files':FILES,'omitted_large_payloads':OMITTED,'frozen_index_sha256':sha((F/'index.json').read_bytes()),'frozen_file_count':len(index['files']),'frozen_bytes':index['total_bytes'],'unique_compiler_dependency_and_link_readbacks':len(deps),'full_local_saved_case_readback':'117 files/all7896cases; compact archive explicitly omits13largepayloads and does not recompute absent numerical cases.'}
finite(catalog);data=(json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode()
DEST.mkdir(parents=True)
for target,value in PENDING.items():
 out=DEST/target;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(value);assert out.read_bytes()==value
(DEST/'catalog.json').write_bytes(data)
print(json.dumps({'files':len(FILES),'bytes':sum(v['bytes'] for v in FILES.values()),'omitted_payloads':len(OMITTED),'catalog_sha256':sha(data),'unique_build_inputs':len(deps)}))
