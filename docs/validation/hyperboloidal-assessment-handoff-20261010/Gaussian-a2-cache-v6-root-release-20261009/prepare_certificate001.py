"""SOURCE-ONLY prepared metadata admission; root must read then execute once."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OWNER=BASE/'continuum/manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009'
ADD=BASE/'continuum/Gaussian-a2-cache-v6-inventory-addendum-20261009'
INDEPENDENT=BASE/'boundary/manufactured-Gaussian-a2-cache-v6-independent-source-review-20261009'
INDEX_SHA='cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397'
RECIPE_SHA='9a7ee237b6fd4d764bac38e62ff4307248fbc4beb57b7a0dbd307e690d0e0742'
INDEPENDENT_INDEX_SHA='b6d678bd8ee486fb9895b3400f149b3c74f967bd092ccfa3aa787c828dbc8388'
INDEPENDENT_RECEIPT_SHA='a110a702faeea2d73338034e0c3aed9d83e2dde4b9338c22ac2c7d015f0f9aca'
EXPECTED_LAUNCH_SHA='ee69a068fb4ef2ab9235022ebf2bc06316c725c189f86c8d76fe3e23fd1e2986'
EXISTING_ROOT_HISTORY={
 'review_source001.py':'f83c044bb4ad07b87962697cadea42273cc601ec52bfa99f8bbacdda48be66ac',
 'review_source002.py':'547ef2b9928bc60780624df6f3dc4559cec357a2ff43e3ea634aadddaf0b701b',
 'source-pins002.json':'15190ec460a653afc492c38d2d8b4d313c8d12e6be620da18eac6869c6e6141b',
 'source-review002.json':'84aeccdca882ed14b26e268a10244cf93d7fa1b09b0e1460191c47ccb4adb0ce',
 'static-review001-failure.json':'72ade0124d587f6d8c8a5394ff3a84f0d13b98a93d2ed4d7b51d2548c64ccd33'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def write(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def require(value,message):
 if not value:raise RuntimeError(message)
def verify(pins):
 for name,digest in pins.items():require(sha(name)==digest,'changed protected input '+name)
def add_pin(pins,path,digest=None):
 p=Path(path).absolute();h=sha(p) if digest is None else digest
 require(str(p) not in pins or pins[str(p)]==h,'conflicting protected pin '+str(p))
 require(sha(p)==h,'protected pin mismatch '+str(p));pins[str(p)]=h

def main():
 invocation=HERE/'preparation-invocation001';invocation.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();pins={};before={};record={'prepared':False,'no_interval_arithmetic':True,
      'candidate_imported':False,'certificate_executed':False,'replay_authorized':False}
 try:
  require(sys.flags.optimize==0 and sys.flags.isolated==1 and sys.dont_write_bytecode,'exact unoptimized -I -B metadata preparation route')
  output=OWNER/'attempts/certificate001';outer_output=HERE/'certificate-invocation001'
  for p in (output,outer_output,HERE/'certificate-authorization.json',HERE/'certificate-release.json',HERE/'certificate-source-review001.json'):
   require(not p.exists(),'fresh one-shot destination required '+str(p))
  for name,h in EXISTING_ROOT_HISTORY.items():add_pin(pins,HERE/name,h)
  prior_root=BASE/'Gaussian-a2-interval-cache-v5-root-release-20261009'
  add_pin(pins,prior_root/'prepare_certificate001.py','f5fbc2151dd1656b6e87643d0a9aa97e183520961c1750bf229637736ae4e5c9')
  add_pin(pins,prior_root/'launch.py','f0581b013786e812cae83a563059fd7c84f4d697e2da3e0d56b6e7937bb925e2')
  require(sha(OWNER/'source-index.json')==INDEX_SHA and sha(OWNER/'recipe.json')==RECIPE_SHA,'exact reviewed v6 index/recipe')
  require(sha(INDEPENDENT/'index.json')==INDEPENDENT_INDEX_SHA and sha(INDEPENDENT/'receipt.json')==INDEPENDENT_RECEIPT_SHA,'exact independent review')
  root_review=load(HERE/'source-review002.json');independent=load(INDEPENDENT/'receipt.json')
  require(root_review.get('passed') is True and root_review.get('root_instrumentation_source_review_complete') is True
      and root_review.get('source_index_sha256')==INDEX_SHA and root_review.get('protected_inputs')==237
      and root_review.get('inputs_unchanged') is True,'actual root review002 scope')
  require(independent.get('passed') is True and independent.get('reviewed_source_index_sha256')==INDEX_SHA
      and independent.get('inputs_unchanged') is True and independent.get('prerequisite_combined_units')==179
      and independent.get('replay_admitted') is False,'actual independent instrumentation review')
  reviewed_pins=load(HERE/'source-pins002.json');require(len(reviewed_pins)==237,'exact root237 input inventory')
  for name,h in reviewed_pins.items():add_pin(pins,name,h)
  recipe=load(OWNER/'recipe.json');index=load(OWNER/'source-index.json')
  require(sha(ADD/'index.json')=='62be282de52536b2e5f3f5c82bf43f9de309efcbe3e38ab4b146325717aac4f2','inventory erratum index')
  addendum=load(ADD/'additional-protected-history.json')
  require(sha(ADD/'additional-protected-history.json')=='67253feac124033dd249568d89e62f70fab0396b0b2333dd3258c37aef6307c2','three history indices erratum')
  for row in index['files']+recipe['protected_inputs']+addendum['files']+[recipe['python']]:
   require(Path(row['path']).stat().st_size==row['bytes'],'protected size '+row['path']);add_pin(pins,row['path'],row['sha256'])
  add_pin(pins,OWNER/'source-index.json',INDEX_SHA)
  for folder in (ADD,INDEPENDENT):
   for p in folder.rglob('*'):
    if p.is_file():add_pin(pins,p)
  science=('interval.py','interval_uncached.py','producer_bounds.py','replay_bounds.py','replay_stage.py','unit_stage.py','cache_units.py','run_once.py')
  for name in science:
   current=(OWNER/name).read_bytes();prior=(OWNER/'history/v5'/name).read_bytes()
   require(current==prior and ast.dump(ast.parse(current),include_attributes=False)==ast.dump(ast.parse(prior),include_attributes=False),'unchanged math body '+name)
  observed=(OWNER/'certificate_stage.py').read_text()
  erased=re.sub(r'^[ \t]*# BEGIN_OBSERVATION_ONLY\n.*?^[ \t]*# END_OBSERVATION_ONLY\n','',observed,flags=re.M|re.S)
  prior=(OWNER/'history/v5/certificate_stage.py').read_text()
  require(erased==prior and ast.dump(ast.parse(erased),include_attributes=False)==ast.dump(ast.parse(prior),include_attributes=False),'observation-only producer equality')
  require(recipe['domain_wall_seconds']==600 and recipe['stage_timeouts']['certificate']==660,'fixed600/660 limits')
  require(recipe['bits']==256 and recipe['replay_bits']==384 and recipe['series_order']==32
      and recipe['max_depth']==60 and recipe['max_leaves']==1048576 and recipe['max_certificate_bytes']==67108864
      and recipe['cache_endpoint_cap']==4096 and recipe['a']=='2' and recipe['sigma']==['7/20','1/2']
      and recipe['epsilon_endpoint']=='3/4' and recipe['instrumentation_cadence_nodes']==128
      and recipe['no_resume'] is True and recipe['replay_admitted_by_this_candidate'] is False,'unchanged mathematical/DFS/cache/domain bounds')
  units=recipe['cached_v3_units_receipt'];require(units['sha256']=='347cfc3c1397a1d9e1a79a8951d0d5336cb73e2853f28bb54ec16d732ec53eaf','actual179 unit receipt')
  u=load(units['path']);report=load(recipe['cached_v3_units_report']['path']);cache=load(recipe['cached_v3_cache_report']['path'])
  require(u.get('completed') is True and u.get('passed') is True and type(u.get('returncode')) is int
      and u['returncode']==0 and u.get('inputs_unchanged') is True and u.get('stage')=='units'
      and u.get('source_index_sha256')==recipe['cached_v3_source_index_sha256']
      and u.get('recipe_sha256')==recipe['cached_v3_recipe_sha256'],'successful exact same-v3 unit source binding')
  require(report.get('passed') is True and report.get('combined_unit_count')==179 and report.get('case_count')==144
      and report.get('cache_unit_count')==35 and report.get('domain_boxes_evaluated')==0
      and report.get('cache_units_passed') is True and cache.get('passed') is True and cache.get('case_count')==35
      and len(cache.get('cases',[]))==35 and all(x.get('passed') is True for x in cache['cases']),'actual144+35 unit reports')
  for p in (recipe['cached_v3_units_report'],recipe['cached_v3_cache_report']):require(p in u['outputs'],'unit receipt report hash binding')
  for version,expected in (('v4','17bc02765f47f9880c4f5ebfd9026748c316cc0a8133ab954c54042c32b7c0c0'),('v5','46bbdd2c0883b0f64d951cdb1152a8bc4de5fad6239d7a7d266b02d476626156')):
   entry=recipe['cached_'+version+'_timeout_receipt'];add_pin(pins,entry['path'],expected);failed=load(entry['path'])
   require(entry['sha256']==expected and failed.get('completed') is False and failed.get('passed') is False
      and type(failed.get('returncode')) is int and failed['returncode']==1 and failed.get('inputs_unchanged') is True
      and failed.get('stage')=='certificate' and failed.get('source_index_sha256')==recipe['cached_'+version+'_source_index_sha256']
      and failed.get('recipe_sha256')==recipe['cached_'+version+'_recipe_sha256'],'immutable '+version+' actual capped FAIL')
   require(not Path(entry['path']).with_name('report.json').exists(),version+' failed attempt has no report')
   for suffix in ('progress','stderr','command','partial_certificate_metadata'):
    p=recipe['cached_'+version+'_'+suffix];require(p in failed['outputs'],version+' history output bound by receipt')
    add_pin(pins,p['path'],p['sha256'])
   require('UNRESOLVED: declared domain time limit' in Path(recipe['cached_'+version+'_stderr']['path']).read_text(),'actual '+version+' cap cause')
  require(sha(HERE/'launch.py')==EXPECTED_LAUNCH_SHA,'reviewed new process-group launcher source')
  for name in ('launch.py','prepare_certificate001.py','PREPARED-PLAN.md','launcher-source-index001.json'):
   add_pin(pins,HERE/name)
  launcher_source_index=load(HERE/'launcher-source-index001.json')
  for row in launcher_source_index['files']:add_pin(pins,row['path'],row['sha256'])
  # Additional launcher stdlib context is metadata only, never scientific input.
  stdlib=Path(recipe['python']['path']).parent.parent/'lib/python3.9'
  for name in ('signal.py','subprocess.py','os.py','pathlib.py','argparse.py','hashlib.py'):
   add_pin(pins,stdlib/name)
  verify(pins);before={name:sha(name) for name in pins};write(invocation/'pins-before.json',before)
  review_record={'passed':True,'root_source_review002_sha256':sha(HERE/'source-review002.json'),
      'independent_review_index_sha256':INDEPENDENT_INDEX_SHA,'independent_review_receipt_sha256':INDEPENDENT_RECEIPT_SHA,
      'root237_pins_retained':True,'inventory_addendum_included':True,'actual179_units_retained':True,
      'actual_v4_v5_capped_FAIL_preserved':True,'eight_math_bodies_byte_AST_equal_v5':True,
      'producer_observation_removal_byte_AST_equal_v5':True,'certificate_only':True,'replay_authorized':False,
      'caps':{'producer':600,'inner_wrapper':660,'root_process_group':720},'no_resume':True,
      'partial_payloads_stream_hash_only':True,'no_interval_or_domain_evaluation_by_preparation':True}
  write(HERE/'certificate-source-review001.json',review_record);add_pin(pins,HERE/'certificate-source-review001.json')
  auth={'allow_execution':True,'stage':'certificate','source_index_sha256':INDEX_SHA,'recipe_sha256':RECIPE_SHA,
      'output':str(output),'units_receipt':units,'prior_timeout_receipt':recipe['cached_v4_timeout_receipt'],
      'prior_v5_timeout_receipt':recipe['cached_v5_timeout_receipt'],
      'root_source_review002_sha256':sha(HERE/'source-review002.json'),'independent_source_review':{'path':str(INDEPENDENT/'receipt.json'),'sha256':INDEPENDENT_RECEIPT_SHA},
      'scope':'One fresh instrumentation-only600/660s diagnostic producer with720s root process-group cap; no replay/resume/regional or global acceptance'}
  write(HERE/'certificate-authorization.json',auth)
  release={'owner':str(OWNER),'pins':pins,'authorization_sha256':sha(HERE/'certificate-authorization.json'),
      'output':str(output),'stage':'certificate','source_index_sha256':INDEX_SHA,'recipe_sha256':RECIPE_SHA,
      'caps':{'producer':600,'inner_wrapper':660,'root_process_group':720},'independent_review_receipt_sha256':INDEPENDENT_RECEIPT_SHA}
  write(HERE/'certificate-release.json',release)
  verify(pins)
  record.update(prepared=True,pins=len(pins),authorization_sha256=sha(HERE/'certificate-authorization.json'),
      release_sha256=sha(HERE/'certificate-release.json'),actual179_units_retained=True,prior_v4_v5_failures_retained=True)
 except BaseException as error:
  record.update(exception_type=type(error).__name__,exception=str(error));(invocation/'failure.txt').write_text(traceback.format_exc())
 finally:
  after={}
  for name in pins:
   try:after[name]=sha(name)
   except OSError as error:after[name]={'error':str(error)}
  write(invocation/'pins-after.json',after)
  record['inputs_unchanged']=bool(before) and all(after.get(name)==digest for name,digest in before.items())
  record['seconds']=time.monotonic()-started;write(invocation/'receipt.json',record)
 print(json.dumps(record),flush=True)
 return 0 if record['prepared'] and record['inputs_unchanged'] else 1

if __name__=='__main__':raise SystemExit(main())
