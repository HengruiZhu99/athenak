"""Standard-library-only one-shot collector preparation; no archive/scientific calls."""
from pathlib import Path
import ast,hashlib,json,subprocess
P=Path(__file__).resolve().parent;R=P.parents[2];B=R/'build-layer-research'
def pin(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'source':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):s=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode();assert len(s)<=1048576;Path(p).write_bytes(s)
def policy(path,size):
 p=Path(path);suffix=p.suffix.lower()
 if size>1048576:return 'file exceeds1MiB; exact original bytes metadata-only'
 if suffix in {'.rst','.bin','.npy','.npz','.jsonl','.o','.obj','.a','.so','.dylib','.dll','.exe','.pyc','.pyo','.h5','.hdf5','.pkl','.pickle'}:return 'raw array/compiled/object/JSONL payload metadata-only'
 with p.open('rb') as f:magic=f.read(8)
 if magic[:4] in {b'\x7fELF',b'\xfe\xed\xfa\xce',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'} or magic.startswith(b'!<arch>'):return 'compiled binary signature metadata-only'
 if magic.startswith(b'\x93NUMPY') or magic.startswith(b'\x89HDF'):return 'raw scientific array signature metadata-only'
 return None
assert not (P/'recipe.json').exists(),'preserve source-only preparation; never overwrite'
destination=R/'docs/validation/hyperboloidal-reference-wave-map-native-t2-N24-controls-20261009';assert not destination.exists()
rootspec=[
 ('partial-half',B/'reference-wave-map-partial-half-N24-held-20261009'),
 ('partial-small',B/'reference-wave-map-partial-small-N24-held-20261009'),
 ('two-partial-launch-and-summary',B/'boundary/reference-wave-map-two-new-N24-partial-launch-20261009'),
 ('two-partial-source-preparation',B/'boundary/reference-wave-map-two-new-N24-partial-preparation-20261009'),
 ('root-manual-two-cases',B/'wave-map-native-two-N24-partial-fields-root-20261009'),
 ('root-scalar-comparison',B/'wave-map-native-N24-partial-comparison-root-20261009'),
 ('root-release-half',B/'wave-map-native-half-N24-partial-root-release-20261009'),
 ('root-release-small',B/'wave-map-native-small-N24-partial-root-release-20261009'),
 ('original-native-half-failed',B/'wave-map-native-t2-root-20261009/batch001/wave-map-half-N24-large-t2'),
 ('original-native-small-failed',B/'wave-map-native-t2-root-20261009/batch001/wave-map-N24-small-t2'),
 ('original-native-C0-completed',B/'wave-map-native-t2-root-20261009/batch001/c0-N24-large-t2'),
 ('completed-C0-readback',B/'reference-wave-map-t2-readback-held-20261009/attempts/c0-N24-large-t2-001'),
 ('completed-C0-root-release',B/'wave-map-native-completed-C0-N24-root-release-20261009'),
 ('savedJSON-comparison-figure',B/'boundary/reference-wave-map-N24-saved-comparison-figure-20261009')]
planned={};roots=[]
def add(p,relative,role):
 item=pin(p);item.update(archive_relative=relative,role=role,omission_reason=policy(p,item['bytes']));assert relative not in planned;planned[relative]=item
for tag,root in rootspec:
 assert root.is_dir();inventory=[]
 for p in sorted(root.rglob('*')):
  if p.is_file():inventory.append(str(p.resolve()));add(p,tag+'/'+str(p.relative_to(root)),tag)
 roots.append({'tag':tag,'source':str(root),'inventory':inventory})
# Selected standard context and completed-wrapper source; never collect their parent trees.
comparison=load(B/'wave-map-native-N24-partial-comparison-root-20261009/recipe.json')
standard=next(x for x in comparison['cases'] if x['tag']=='standard')
for key in ['owner_observations','owner_receipt','manual_rows','manual_receipt','native_stderr']:
 path=Path(standard[key]);add(path,'standard-N24-comparison-context/'+key+path.suffix,'previous completed standard N24 scalar context only')
for name in ['run_t2_readback.py','recipe.json','source-review-index-v2.json','source-only-readiness.json','release-schema.json','PLAN.md']:
 add(B/'reference-wave-map-t2-readback-held-20261009'/name,'completed-wrapper-source-context/'+name,'generic wrapper metadata/source only; no other attempt')
# Parent supplied exact success/failure scope; validate saved gate metadata before planning.
gates=[]
def gate(path,sha,expected):
 x=pin(path);assert x['sha256']==sha,path;j=load(path)
 for k,v in expected.items():assert j[k]==v,(str(path),k)
 gates.append({**x,'expected':expected})
half=B/'reference-wave-map-partial-half-N24-held-20261009/attempts/wave-map-half-N24-large-t2-001'
small=B/'reference-wave-map-partial-small-N24-held-20261009/attempts/wave-map-N24-small-t2-001'
c0=B/'reference-wave-map-t2-readback-held-20261009/attempts/c0-N24-large-t2-001'
gate(half/'receipt.json','39b0492d20328c61b565eb1b5432dc748fbb2ec7c7fcaa7cebfd5af277fcddc3',{'observer_completed':True,'accepted_native_run':False,'native_returncode':-6,'saved_restart_files':32,'protected_before_after_equal':True})
gate(small/'receipt.json','94b640ae8727e3005d8be5423e5df5317f7ffab3d58e4d941a027cd18584349c',{'observer_completed':True,'accepted_native_run':False,'native_returncode':-6,'saved_restart_files':41,'protected_before_after_equal':True})
gate(B/'wave-map-native-N24-partial-comparison-root-20261009/attempt001/report.json','0610afdad653199a50faad87455b241c9450651ba6c760950a0b0e442973fd76',{'passed_scalar_readback':True,'all_saved_independent_field_pairs':105,'accepted_native_run':False})
gate(c0/'receipt.json','4e02b0f677466d03aa126e2283b6cefde5951895aad84896ac901095afb59bfc',{'returncode':0,'passed_completed_t2_snapshot_gates':True,'protected_before_after_equal':True})
gate(c0/'analysis/receipt.json','83925440c809c9e14371f4ef409de221bbebb2ac0e57709d5d086e694fef642b',{'passed_saved_snapshot_finite_and_diagnostic_gates':True,'saved_arrays':81,'N':24,'target_time':2.0,'protected_inputs_before_after_equal':True})
gate(B/'wave-map-native-t2-root-20261009/batch001/c0-N24-large-t2/launch-receipt.json','ba502d2f4fdf17aa359939a2caf1f5ddc63b51dedfa7d8955cb2cc1dc4937f8c',{'returncode':0,'passed_native_process_and_provenance':True,'sources_before_after_equal':True})
for tag,path in [('half',half),('small',small),('C0',c0)]:assert (path/'protected-inputs-before.json').read_bytes()==(path/'protected-inputs-after.json').read_bytes()
pairs=[[str(path/'protected-inputs-before.json'),str(path/'protected-inputs-after.json')] for path in [half,small,c0,c0/'analysis']]
# Union all recorded dependency identities without copying any external payload.
deps={}
map_paths=[half/'protected-inputs-before.json',small/'protected-inputs-before.json',c0/'protected-inputs-before.json',c0/'analysis/protected-inputs-before.json',B/'wave-map-native-two-N24-partial-fields-root-20261009/attempts/wave-map-half-N24-large-t2-001/pins-before.json',B/'wave-map-native-two-N24-partial-fields-root-20261009/attempts/wave-map-N24-small-t2-001/pins-before.json',B/'boundary/reference-wave-map-N24-saved-comparison-figure-20261009/attempt001/pins-before.json']
for path in map_paths:
 x=load(path);items=x.items() if isinstance(x,dict) else ((row['path'],row['sha256']) for row in x)
 for name,digest in items:
  if name in deps:assert deps[name]['sha256']==digest,name
  row=pin(name);assert row['sha256']==digest,name;deps[name]=row
parts=[];(P/'dependency-input-parts').mkdir(exist_ok=False)
for i in range(0,len(deps),256):
 path=P/f'dependency-input-parts/part-{i//256:03d}.json';dump(path,{'scope':'Exact recorded external source/compiler/runtime/executable/object/array identities only. No payload is copied during preparation.','inputs':list(sorted(deps.values(),key=lambda x:x['source']))[i:i+256]});parts.append(pin(path))
recipe={'source_only_preparation':True,'execution':'HELD','destination':str(destination),'roots':roots,'planned_files':list(planned.values()),'dependency_input_manifests':parts,'external_dependency_records':len(deps),'recorded_dependency_maps':[pin(p) for p in map_paths],'completion_gates':gates,'identical_inventory_pairs':pairs,'production_reference_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2','policy':{'max_copied_bytes_per_file':1048576,'all_arrays_executables_objects_NPY_NPZ_JSONL_metadata_only':True,'binary_signatures_checked':True,'logs_exact_bytes_whitespace_preserved':True,'large_logs_not_truncated_or_normalized':True,'never_overwrite_or_repeat_existing_destination':True,'no_live_parent_batch_or_values_or_bulk_tree':True},'scope':'One compact completed N24 controls checkpoint; wave-map nativefailures remain partial, C0 completed81 savedgates; no scientific calls or stability inference.'}
dump(P/'recipe.json',recipe)
files=[pin(P/name) for name in ['collect_once.py','prepare_metadata.py','PLAN.md','release-schema.json','recipe.json']]+parts
for p in [P/'collect_once.py',Path(__file__).resolve()]:ast.parse(p.read_text(),filename=str(p))
dump(P/'source-index.json',{'source_only':True,'execution':'HELD','files':files,'scope':recipe['scope'],'archive_destination_created':False,'scientific_imports_or_calls':0})
for row in recipe['planned_files']+list(deps.values()):assert pin(row['source'])=={k:row[k] for k in ('source','sha256','bytes')}
ready={'source_only':True,'execution':'HELD','source_index':pin(P/'source-index.json'),'recipe':pin(P/'recipe.json'),'collector':pin(P/'collect_once.py'),'planned_source_roots':len(roots),'planned_files':len(planned),'planned_copied_files':sum(x['omission_reason'] is None for x in planned.values()),'planned_metadata_only_payloads':sum(x['omission_reason'] is not None for x in planned.values()),'recorded_external_dependency_inputs':len(deps),'dependency_manifest_parts':len(parts),'completed_C0_wrapper_and81_snapshot_gates_verified':True,'two_partial_observers_and_root105state_comparison_verified':True,'no_archive_created':not destination.exists(),'no_scientific_imports_queries_arrays_or_native':True}
dump(P/'source-only-readiness.json',ready);print(json.dumps(ready,indent=2))
