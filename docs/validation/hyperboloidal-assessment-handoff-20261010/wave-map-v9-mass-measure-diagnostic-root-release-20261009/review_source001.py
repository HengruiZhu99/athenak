"""Static source/operand review; no array decoding or target arithmetic."""
from pathlib import Path
import hashlib,json,ast,textwrap
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
SRC=BASE/'boundary/reference-wave-map-v9-mass-measure-diagnostic-held-20261009'
OLD=BASE/'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v9-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
idx=SRC/'source-index.json';assert sha(idx)=='c8d1b760ec7dea2711719078e434139af28496f5050c3ea7a336d6e39ff2103f'
pins=load(SRC/'input-pins.json')
for row in load(idx)['files']:
 assert row['path'] not in pins or pins[row['path']]==row['sha256']
 pins[row['path']]=row['sha256']
pins[str(idx)]=sha(idx)
for p,h in pins.items():assert sha(p)==h,p
recipe=load(SRC/'recipe.json');assert sha(SRC/'recipe.json')=='66b29ff7f20668bce2bcdee035dc97900ab95f565e8bd2dee2fd8c174ddd7e78'
assert recipe['first_radius']==609 and recipe['last_radius']==640 and recipe['expected_components']==131072
assert recipe['selected_npz_arrays']==['source_coefficient_radii','radial_weights','angular_weights','source_reference_rows']
assert (SRC/'tiny_normalization.py').read_bytes()==(OLD/'tiny_normalization.py').read_bytes()
assert (SRC/'fast_weighting.py').read_bytes()==(OLD/'fast_weighting.py').read_bytes()
old=(OLD/'verify_retained.py').read_text();new=(SRC/'operand_graph.py').read_text()
for name in ('mm','bilinear','action','jacobi_jet','modal_jet'):
 a=[n for n in ast.walk(ast.parse(old)) if isinstance(n,ast.FunctionDef) and n.name==name]
 b=[n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.FunctionDef) and n.name==name]
 assert len(a)==len(b)==1
 aa=textwrap.dedent(ast.get_source_segment(old,a[0]));bb=textwrap.dedent(ast.get_source_segment(new,b[0]))
 assert aa==bb and ast.dump(a[0],include_attributes=False)==ast.dump(b[0],include_attributes=False),name
failure=load(Path(recipe['failed_receipt']));assert not failure['completed'] and failure['returncode']==1 and failure['inputs_unchanged']
progress=load(Path(recipe['failed_progress']));assert progress['radius_index']==608 and progress['total_radii']==769
for p,h in pins.items():assert sha(p)==h,p
write(HERE/'source-pins001.json',pins)
result={'root_source_math_and_admission_review_passed':True,'independent_review_required_before_execution':True,'source_index_sha256':sha(idx),'protected_inputs':len(pins),'inputs_unchanged':True,'copied_operand_functions_byte_and_AST_equal':True,'tiny_and_weighting_helpers_unchanged':True,'selected_NPZ_arrays':recipe['selected_npz_arrays'],'read_only_map_window':[609,640],'fixed_product_count':131072,'no_arrays_or_targets_decoded':True,'candidate_imports':False,'no_SVD_or_E_accumulation_or_queries':True,'actual_v9_radial_remains_failed':True,'exact_failure_onset_unknown':True,'review_scope':'Exact saved operand graph and conservative tiny-product observation only; local rounding bounds do not qualify the matrix','read_order':'Text source/PLAN read before capture; every supplied source/input hash reverified twice in this static gate'}
write(HERE/'source-review001.json',result);print(json.dumps(result))
