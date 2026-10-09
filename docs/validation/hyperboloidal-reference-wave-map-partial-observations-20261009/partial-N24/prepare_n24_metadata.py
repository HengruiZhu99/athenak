"""Source and failed-receipt metadata only; no arrays/history/probe calls."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'build-layer-research/reference-wave-map-partial-diagnostic-held-20261009'
CASE='wave-map-N24-large-t2'
FAILED=ROOT/'build-layer-research/wave-map-native-t2-root-20261009/batch001'/CASE/'launch-receipt.json'
FSHA='5375335b4df19137389a81d6f932b2a6cb567dc99157a946246836665c9f292a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (HERE/'recipe.json').exists()
assert sha(OLD/'observe_partial.py')=='c7ada0324bcd91c1e359620f46efd6614286876810a9d692a98ef6e37cb3a0f3'
assert sha(OLD/'recipe.json')=='070be42ffaf40f6d8eeba5b35908b143b2a6f74289ceaf376f02ea232f946e93'
assert sha(FAILED)==FSHA
old=load(OLD/'recipe.json');f=load(FAILED)
assert f['returncode']==-6 and f['passed_native_process_and_provenance'] is False
assert f['sources_before_after_equal'] is True
assert f['stderr_sha256']=='bc0dfe0f95d5e7e10c5f36b19bd5b6062c058837d7553d845010b2c334f93aa0'
spec=old['cases'][CASE]
assert f['input_path']==spec['input_path'] and f['mode']==spec['mode']
ctx=HERE/'source-context';ctx.mkdir(exist_ok=False)
copy_sources={'original-observer.py':OLD/'observe_partial.py','original-N16-recipe.json':OLD/'recipe.json','original-failed-N24-launch.json':FAILED,'N24-exact-input.athinput':Path(spec['input_path'])}
copies={}
for name,src in copy_sources.items():
 dst=ctx/name;dst.write_bytes(src.read_bytes());copies[str(dst)]={'source':str(src),'sha256':sha(src)}
for name in ['observe_partial.py','release-schema.json']:(HERE/name).write_bytes((OLD/name).read_bytes())
compile((HERE/'observe_partial.py').read_text(),str(HERE/'observe_partial.py'),'exec')
fixed=dict(old['fixed_pins'])
fixed.update({str(OLD/'recipe.json'):sha(OLD/'recipe.json'),str(OLD/'observe_partial.py'):sha(OLD/'observe_partial.py'),str(FAILED):FSHA})
for p in [HERE/'PLAN.md',HERE/'prepare_n24_metadata.py',HERE/'release-schema.json']:
 fixed[str(p)]=sha(p)
for p,item in copies.items():fixed[p]=item['sha256']
for p,digest in fixed.items():assert sha(p)==digest,p
r=dict(old)
r.update(prepared_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),cases={CASE:spec},initial_review_failures={CASE:{'original_failed_launch_receipt':str(FAILED),'sha256':FSHA,'returncode':f['returncode'],'output_inventory_files':len(f['outputs'])}},fixed_pins=fixed,source_context_copies=copies,original_unchanged_N16_recipe_sha256=sha(OLD/'recipe.json'),scope='Held stopped failed N24 observation only; exact unchanged c7ada observer/probe/thresholds. No new scientific calls.')
dump(HERE/'recipe.json',r)
ready={'source_only':True,'observer_sha256':sha(HERE/'observe_partial.py'),'observer_byte_equal_to_N16':(HERE/'observe_partial.py').read_bytes()==(OLD/'observe_partial.py').read_bytes(),'recipe_sha256':sha(HERE/'recipe.json'),'plan_sha256':sha(HERE/'PLAN.md'),'original_failed_launch_receipt_sha256':FSHA,'source_context_copies':copies,'output_inventory_files':len(f['outputs']),'saved_rst_inventory_count':sum(p.endswith('.rst') for p in f['outputs']),'original_N16_recipe_unchanged':sha(OLD/'recipe.json')=='070be42ffaf40f6d8eeba5b35908b143b2a6f74289ceaf376f02ea232f946e93','native_arrays_histories_or_logs_read':False,'new_probe_calls':0,'new_native_calls':0,'execution':'HELD'}
dump(HERE/'source-only-readiness.json',ready)
print(json.dumps({'observer_sha256':ready['observer_sha256'],'recipe_sha256':ready['recipe_sha256'],'readiness_sha256':sha(HERE/'source-only-readiness.json'),'rst_metadata_count':ready['saved_rst_inventory_count'],'execution':'HELD'}))
