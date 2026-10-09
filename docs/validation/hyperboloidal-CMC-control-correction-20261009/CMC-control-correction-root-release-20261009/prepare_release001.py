"""Release two separately scoped stages after source/math review; stdlib only."""
from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SRC=REPO/'build-layer-research/continuum/native-angular-pulse-flat-derivatives-CMC-control-correction-held-20261009'
REVIEW=REPO/'build-layer-research/continuum/native-angular-pulse-CMC-control-independent-source-review-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
fixed={'source-index.json':'ba6dcec5efe71a9dd01fa77b98267b384ece508910117c0741a9ad996159f2fe','control-recipe.json':'7e118deaa33c165603271b71cc99a3618ef7c74978591a79c303c78ebfae03f5','derivative_core.py':'8207de6849fd5672213cc0d66e8c632a6457eaa4f0f4fe168f7bdb5e81eaaf47','controls_only.py':'750df1356bedf61260a3e918eb2fd28de303b7f606a1cf9448a90b0734887192'}
for p,h in fixed.items():assert sha(SRC/p)==h,p
assert sha(REVIEW/'index.json')=='03a20cab012659652bec7fadc32d1640662adc5fdd3ea194cf54259f0a9e36d5'
assert sha(REVIEW/'receipt.json')=='8e834eadaa7222eefe6d44cfd14febfd2df69635bc5f3befe32f1d04486d03ea'
rv=load(REVIEW/'receipt.json');assert rv['passed'] and rv['source_bytes_unchanged'] and not rv['remaining_blocking_source_findings'] and not rv['candidate_import_or_execution']
recipe=load(SRC/'control-recipe.json');old=load(recipe['original_full_recipe']['path'])
assert all(old[k]==v for k,v in recipe['scientific_settings'].items())
assert (recipe['fixed_roots'],recipe['fixed_rows'],recipe['fixed_checks'],recipe['saved_other_check_count'])==(189440,100,880,1480)
before=(SRC/'source-history/derivative_core-original.py').read_bytes();after=(SRC/'derivative_core.py').read_bytes()
oldline=b'            b=omega*q/self.a\n';newline=b'            b=omega*(sumjet(t*t for t in Y)**mp.mpf(".5"))/self.a\n'
assert before.count(oldline)==1 and after.count(newline)==1 and after.replace(newline,oldline)==before
assert ast.dump(ast.parse(after.replace(newline,oldline)),include_attributes=False)==ast.dump(ast.parse(before),include_attributes=False)
pins=dict(recipe['protected_pins']);pins.update(recipe['mpmath_python_pins'])
for row in load(SRC/'source-index.json')['files']:
 p=str(SRC/row['path'])
 if p in pins:assert pins[p]==row['sha256']
 pins[p]=row['sha256']
pins[str(SRC/'source-index.json')]=fixed['source-index.json']
for p in REVIEW.rglob('*'):
 if p.is_file():pins[str(p)]=sha(p)
for p,h in pins.items():assert sha(p)==h,p
original=load(recipe['original_full_receipt']);assert original['passed_analytic_scalar_derivative_gate'] is False and original['checks']==2360 and len(original['failed_checks'])==156 and original['source_before']==original['source_after']
for key in ('fresh_control_output','fresh_saved_qualification_output'):assert not Path(recipe[key]).exists()
root_review={'passed':True,'source_reviewed_in_full':True,'single_noncenter_CMC_radius_jet_change':True,'reverse_byte_AST_identity':True,'original_control_loop_targets_and_thresholds_unchanged':True,'fixed_counts':[100,189440,880],'saved_noncontrol_count':1480,'original_full_FAIL_preserved':True,'no_scientific_preparation':True,'independent_review_sha256':sha(REVIEW/'receipt.json'),'protected_pins':len(pins)}
write(HERE/'source-review001.json',root_review)
for stage,key,output in [('controls','corrected_control_gate_execution_admitted','fresh_control_output'),('qualification','saved_check_qualification_execution_admitted','fresh_saved_qualification_output')]:
 auth={key:True,'source_index':{'path':str(SRC/'source-index.json'),'sha256':fixed['source-index.json']},'fresh_output_path':recipe[output],'recipe_sha256':fixed['control-recipe.json'],'root_review_sha256':sha(HERE/'source-review001.json'),'independent_review_sha256':sha(REVIEW/'receipt.json'),'scope':stage,'original_full_gate_remains_failed':True}
 write(HERE/(stage+'-authorization.json'),auth)
write(HERE/'release.json',{'owner':str(SRC),'pins':pins,'authorizations':{stage:sha(HERE/(stage+'-authorization.json')) for stage in ('controls','qualification')}})
print(json.dumps({'prepared':True,'pins':len(pins),'controls_authorization_sha256':sha(HERE/'controls-authorization.json'),'qualification_authorization_sha256':sha(HERE/'qualification-authorization.json')}))
