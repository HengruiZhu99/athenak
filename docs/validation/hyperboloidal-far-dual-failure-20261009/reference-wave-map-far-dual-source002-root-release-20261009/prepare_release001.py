"""Fresh complete field-dual local gate release; preparation imports stdlib only."""
from pathlib import Path
import argparse,hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REPO=BASE.parent
OWNER=BASE/'boundary/reference-wave-map-far-dual-source002-held-20261009'
OUTER=BASE/'boundary/reference-wave-map-outer-arithmetic-source001-held-20261009'
HISTORY=BASE/'boundary/reference-wave-map-far-dual-source001-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-index-sha256',required=True);ap.add_argument('--review-receipt-sha256',required=True);a=ap.parse_args()
review=a.review.resolve();assert sha(review/'index.json')==a.review_index_sha256 and sha(review/'receipt.json')==a.review_receipt_sha256
rv=load(review/'receipt.json');assert rv['passed'] is True
indexsha='79aa9a75b26687d97da135eacf8efc3b577f52c21726304db0c8f17834b6a1e5'
assert sha(OWNER/'source-index.json')==indexsha
assert sha(OWNER/'index.json')=='dfcfd0acb8cf5941095098f60ea90ea6bf7f220bf1ca2db3e773fd7c179b797e'
index=load(OWNER/'source-index.json');recipe=load(OWNER/'recipe.json')
for name in ('probe.cpp','oracle.py','run_once.py'):assert (OWNER/name).read_bytes()==(HISTORY/name).read_bytes(),name
for name in ('reference_wave_map.hpp','reference_wave_map_legacy.hpp','arithmetic_traits.hpp'):
 assert (OWNER/'inputs'/name).read_bytes()==(OUTER/'inputs'/name).read_bytes(),name
assert recipe['fixed_counts']=={'FD_representatives':16,'FD_side_evaluations':160,'base_dual_rows':1904,'bases':112,'closed_positive_dual_rows':18,'legacy_negative_reuse_rows':3,'physical_reference_dual_rows':2352,'positive_dual_rows':2370,'records':2373,'total_helper_evaluations':4900,'zero_gradient_variant_rows':448}
assert recipe['precision_decimal_digits']==[480,560]
assert recipe['thresholds']['native_entrywise_scaled']=='2e-10' and recipe['thresholds']['precision_entrywise_scaled']=='1e-220'
assert recipe['thresholds']['input_seed_formula_nonzero_relative']=='2e-14' and recipe['thresholds']['input_seed_formula_zero_absolute']=='0'
assert sha(OWNER/'inputs/plan/index.json')=='1da6c8208103786560bc7c9ec73c8ce3bbdfeafdbbf0eb6d5e4c6abb44a1af00'
saved=load(BASE/'boundary/reference-wave-map-outer001-saved-readback-v2-20261009/attempt001/receipt.json')
assert saved['passed'] is True and saved['completed'] is True and saved['inputs_unchanged'] is True and saved['returncode']==0
pins={r['path']:r['sha256'] for r in load(recipe['input_pins'])}
for r in index['files']:
 if r['path'] in pins:assert pins[r['path']]==r['sha256']
 pins[r['path']]=r['sha256']
pins[str(OWNER/'source-index.json')]=indexsha
for p in review.rglob('*'):
 if p.is_file():pins[str(p)]=sha(p)
for p,h in pins.items():assert sha(p)==h,p
for name in recipe['attempt_names'].values():assert not (OWNER/'attempts'/name).exists()
assert subprocess.run(['git','diff','--quiet','27c19d20696ea6dd4704032c51dfd026218f64f2','--','src','CMakeLists.txt'],cwd=REPO).returncode==0
write(HERE/'source-review001.json',{'passed':True,'root_full_source_math_review':True,'protected_pins':len(pins),'independent_review_sha256':a.review_receipt_sha256,'complete_physical_P_literal_MP_dual_oracle':True,'cofactor_inverse_independent':True,'all_registered_field_seeds_and_zero_gradient_variants_retained':True,'closed_Fraction_targets_and_legacy_negative_controls':True,'original_State_CastPoint_and_outer_helper_unchanged':True,'source001_metadata_failure_preserved':True,'original_main_source003_FAIL_preserved':True,'FD_five_fixed_levels_and_twelve_components':True,'no_scientific_preparation':True,'production_unchanged':True})
auth={'far_complete_dual_local_execution_admitted':True,'source_index_sha256':indexsha,'recipe_sha256':sha(OWNER/'recipe.json'),'allowed_builds':['release','debug'],'root_review_sha256':sha(HERE/'source-review001.json'),'independent_review_sha256':a.review_receipt_sha256,'scope':'One fixed2373-record/4900-call private local Release then conditional ASan-UBSan Debug; no native or evolution admission'}
write(HERE/'authorization.json',auth)
with (HERE/'launch.py').open('x') as f:f.write((BASE/'outer-reference-wave-map-arithmetic-source001-root-release-20261009/launch.py').read_text())
pins[str(HERE/'launch.py')]=sha(HERE/'launch.py');pins[str(Path(__file__).resolve())]=sha(Path(__file__))
write(HERE/'release.json',{'owner':str(OWNER),'pins':pins,'authorization_sha256':sha(HERE/'authorization.json')})
print(json.dumps({'prepared':True,'pins':len(pins),'authorization_sha256':sha(HERE/'authorization.json')}))
