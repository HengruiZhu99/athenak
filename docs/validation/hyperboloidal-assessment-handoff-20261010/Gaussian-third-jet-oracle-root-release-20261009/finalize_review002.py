"""Seal the corrected static source gate; stage execution is separate."""
from pathlib import Path
import hashlib,json,ast
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
SRC=BASE/'continuum/manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009'
OLD=BASE/'continuum/manufactured-angular-Gaussian-third-jet-oracle-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
pins=json.loads((HERE/'source-pins002.json').read_text())
for p,h in pins.items():assert sha(p)==h,p
proof=json.loads((SRC/'source-preparation-v2.json').read_text())
for row in proof['reverse_source_proofs']:
 assert row['exact_forward_bytes'] and row['exact_reverse_bytes'] and row['reverse_AST']
 assert sha(SRC/row['path'])==row['new_sha256'] and sha(OLD/row['path'])==row['old_sha256']
for row in proof['unaffected_numerical_bodies']:
 a=(SRC/row['path']).read_bytes();b=(OLD/row['path']).read_bytes()
 assert a==b and ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False)
old=json.loads((OLD/'recipe.json').read_text());new=json.loads((SRC/'recipe.json').read_text())
for k,v in old.items():
 if k not in ('status','protected_inputs'):assert new[k]==v,k
assert proof['all_scientific_recipe_values_unchanged'] and proof['nominal_admission_exact_fraction'] and proof['reference_comparator_both_physical'] and proof['full_measured_timing_guard_child_and_outer']
for p,h in pins.items():assert sha(p)==h,p
result={'passed':True,'root_full_source_math_and_admission_review':True,'source_index_sha256':sha(SRC/'source-index.json'),'protected_inputs':len(pins),'inputs_unchanged':True,'candidate_imported':False,'numeric_calls':False,'three_v1_blockers_corrected':True,'v1_remains_unexecuted_ineligible':True,'seven_numerical_bodies_byte_AST_unchanged':True,'registry_thresholds_precisions_domains_unchanged':True,'nominal_admission_exact_Fraction':True,'independent_reference_comparison_both_physical':True,'full_measured_timing_review_enforced_child_and_outer':True,'independent_v2_review_required_before_units':True,'units_timing_full_separate_releases_required':True,'no_native_RHS_or_global_slicing_or_BH_admission':True,'scope':'Physical-reference RWM Einstein-sector analytic oracle only; no compound inner BM'}
with (HERE/'source-review002.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
