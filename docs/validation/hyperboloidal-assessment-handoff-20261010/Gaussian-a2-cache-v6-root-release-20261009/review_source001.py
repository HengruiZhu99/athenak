"""Review the instrumentation successor; no interval module import/evaluation."""
from pathlib import Path
import hashlib,json,ast,re
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
SRC=BASE/'continuum/manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009'
ADD=BASE/'continuum/Gaussian-a2-cache-v6-inventory-addendum-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
index=SRC/'source-index.json';assert sha(index)=='cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397'
add=ADD/'additional-protected-history.json';assert sha(add)=='62be282de52536b2e5f3f5c82bf43f9de309efcbe3e38ab4b146325717aac4f2'
recipe=load(SRC/'recipe.json');pins={str(index):sha(index),str(add):sha(add)}
for row in load(index)['files']+recipe['protected_inputs']+load(add)['files']:
 p,h=row['path'],row['sha256'];assert Path(p).stat().st_size==row['bytes']
 assert p not in pins or pins[p]==h,p
 pins[p]=h
for p in ADD.rglob('*'):
 if p.is_file():pins[str(p)]=sha(p)
for p,h in pins.items():assert sha(p)==h,p
source_paths={str(p) for p in SRC.rglob('*') if p.is_file()}
assert source_paths <= set(pins),source_paths-set(pins)
assert recipe['domain_wall_seconds']==600 and recipe['stage_timeouts']['certificate']==660
assert recipe['bits']==256 and recipe['replay_bits']==384 and recipe['series_order']==32 and recipe['expected_combined_unit_count']==179
assert recipe['max_depth']==60 and recipe['max_leaves']==1048576 and recipe['max_certificate_bytes']==67108864
math_files=['interval.py','interval_uncached.py','producer_bounds.py','replay_bounds.py','replay_stage.py','unit_stage.py','cache_units.py','run_once.py']
for name in math_files:
 a=(SRC/name).read_bytes();b=(SRC/'history/v5'/name).read_bytes()
 assert a==b and ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False),name
text=(SRC/'certificate_stage.py').read_text()
erased=re.sub(r'^[ \t]*# BEGIN_OBSERVATION_ONLY\n.*?^[ \t]*# END_OBSERVATION_ONLY\n','',text,flags=re.M|re.S)
old=(SRC/'history/v5/certificate_stage.py').read_text()
assert erased==old and ast.dump(ast.parse(erased),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
units=load(Path(recipe['cached_v3_units_report']['path']));assert units['passed'] and units['combined_unit_count']==179
prior=load(Path(recipe['cached_v5_timeout_receipt']['path']));assert not prior['passed'] and not prior['completed'] and prior['returncode']==1 and prior['inputs_unchanged']
assert not Path(recipe['cached_v5_timeout_receipt']['path']).with_name('report.json').exists()
assert 'UNRESOLVED: declared domain time limit' in Path(recipe['cached_v5_stderr']['path']).read_text()
for p,h in pins.items():assert sha(p)==h,p
write(HERE/'source-pins001.json',pins)
result={'passed':True,'root_instrumentation_source_review_complete':True,'independent_review_required_before_execution':True,'source_index_sha256':sha(index),'inventory_addendum_sha256':sha(add),'protected_inputs':len(pins),'inputs_unchanged':True,'mathematical_sources_byte_AST_equal_to_v5':math_files,'marked_producer_observation_removal_byte_AST_equal':True,'cache_semantics':'dict membership returns original boolean; deletion returns or raises before its counter; inherited lookup/set/iteration preserve FIFO','progress_scope':'between nodes; exact pending boxes in DFS order, no regional fraction','prior179units_retained':True,'prior_v5_actual_timeout_preserved':True,'fresh600second_diagnostic_only':True,'replay_admitted':False,'candidate_imported':False,'interval_or_target_evaluated':False,'partial_certificate_decoded_or_resumed':False,'global_slicing_accepted':False,'read_order':'Text sources/diffs read before capture; all supplied source/external/history pins reverified twice'}
write(HERE/'source-review001.json',result);print(json.dumps(result))
