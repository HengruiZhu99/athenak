from pathlib import Path
import hashlib,json,subprocess
ROOT=Path('/Users/hz0693/research/hyperboloidal');HERE=Path(__file__).resolve().parent
OWNER=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source003-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/inner-joint-nonlinear-helper-source003-independent-review-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as s:s.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
fixed={'source-index.json':'6d0f91c4b7bf701f83d723f81c89df8265f76fe839ef4f522e6e056b9544d671','recipe.json':'a8b4d9393d9bd745b35dc88f79394dfc2dd2b5b9f0b567fde5658d4c22ca7ab7','inner_gauge.hpp':'b26357cc3d25822b8c2b7f60bdf4bb10163bd904babbbe3231ccf3e5c0aeae4b','probe.cpp':'b6c7a1443206ead9b836413713a38d74188a4ee06f1954ab186345892211269e','oracle.py':'15fd453129045b45348228f6dce66bd2eba2f05daacadae448ffc0f301922160','run_once.py':'8eea82600dcbebda98a8e8e95cbd3427ffdc30b74e40c8084207d0595e666a92'}
for name,d in fixed.items():
 if sha(OWNER/name)!=d:raise RuntimeError('fixed source differs '+name)
if sha(REVIEW/'index.json')!='412dddbbc1c7f77b820b9a4000bf4c17f5c1ea7fa199d908b208cc479007cca9' or sha(REVIEW/'receipt.json')!='c3a4d776e9c106404c7aa4ce6a9248ef71a41ee11a3291dba0a3b7dd83bd902b':raise RuntimeError('review differs')
review=load(REVIEW/'receipt.json')
if not(review['passed_source_math_and_admission_review'] and review['original_inputs_unchanged'] and review['no_candidate_import_AST_syntax_CAS_compile_query_array_decode']):raise RuntimeError('independent review not admitted')
PREVIOUS=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source002-held-20261009'
PRIOR_ROOT=ROOT/'build-layer-research/inner-joint-nonlinear-source002-root-release-20261009'
for name in ['oracle.py','run_once.py']:
 if (PREVIOUS/name).read_bytes()!=(OWNER/name).read_bytes():raise RuntimeError('oracle/runner changed '+name)
for name,d in [('primary-helper.diff','caa2e5a13b67aaaf67247df75e3f4ae08cf8efcbd03a907458357e45f92f53f1'),('audit-only-probe.diff','fc2a38781392a848b06cf393979903d0b3eafaf3ffcc607e38b0cf60a191f32d')]:
 if sha(OWNER/name)!=d:raise RuntimeError('reviewed diff differs '+name)
failure=PREVIOUS/'attempts/Release001/receipt.json'
if sha(failure)!='e89496d60e6055d25c0f969d9f9f4724fb42d50b2df80225f4e68c990ea6e86d' or load(failure)['completed'] or load(failure)['returncode']!=1:raise RuntimeError('prior oracle failure differs')
previous_recipe=load(PREVIOUS/'recipe.json');new_recipe=load(OWNER/'recipe.json')
for key in ['expected_record_counts','thresholds','precision','modes','raw22_order','gauge_raw22_indices','geometry_raw22_indices','high_contrast_families']:
 if previous_recipe[key]!=new_recipe[key]:raise RuntimeError('fixed science settings changed '+key)
r=load(OWNER/'recipe.json');index=load(OWNER/'source-index.json');pins={x['path']:x['sha256'] for x in load(r['input_pins'])}
for row in index['files']:pins[row['path']]=row['sha256']
pins[str(OWNER/'source-index.json')]=fixed['source-index.json']
for p in REVIEW.rglob('*'):
 if p.is_file():pins[str(p)]=sha(p)
for p in PRIOR_ROOT.rglob('*'):
 if p.is_file():pins[str(p)]=sha(p)
for p,d in pins.items():
 if sha(p)!=d:raise RuntimeError('input differs '+p)
principal=load(r['principal_receipt'])
if not(principal['completed'] and principal['passed'] and principal['returncode']==0 and principal['inputs_unchanged']):raise RuntimeError('actual118/exact18 principal gate prerequisite failed')
production=subprocess.run(['git','diff','--quiet','27c19d20696ea6dd4704032c51dfd026218f64f2','--','src','CMakeLists.txt'],cwd=ROOT,check=False)
if production.returncode!=0:raise RuntimeError('production baseline changed')
for name in r['attempt_names'].values():
 if (OWNER/'attempts'/name).exists():raise RuntimeError('single-use output exists')
root_review=dict(passed=True,verified_unique_pins=len(pins),root_full_source_and_math_review=True,source003_three_difference_regimes=True,source003_probe_only_six_branch_audit_fields=True,prior_source002_actual_oracle_failure_preserved=True,prior_source001_compile_failure_preserved=True,prior_root_review_sha256=sha(PRIOR_ROOT/'source-review001.json'),independent_review_sha256=sha(REVIEW/'receipt.json'),production_identical_to='27c19d20696ea6dd4704032c51dfd026218f64f2',
 checks=['complete reference physical connection and grouped literal algebra','P stored with independent Theta, no K/Theta substitution','bounded k variable and relative field-dual derivatives','W1 exact baseline return before inner arithmetic','independent MP split/core witness target; huge A cancellation not claimed at insufficient precision','actual22 geometric rows identity plus independent gauge dual/19 principal coefficients','all three FD levels saved; final-level and convergence/floor gates mandatory','source/input guards and unoptimized isolated oracle before import'],
 limits=['declared finite local families only; no native or puncture/BH/global stability','branch formulas are not universal correctly-rounded sums or arbitrary-positive-state accuracy certificates','unchanged dV and coupled reference-gradient split retain separate simultaneous metric/field contrast risk','extreme coefficient AD checked for relative seeds only; finite q.valid does not certify arbitrary seed accuracy'],no_scientific_import_compile_or_query_by_preparation=True)
write(HERE/'source-review001.json',root_review)
auth=dict(local_nonlinear_helper_execution_admitted=True,recipe_sha256=fixed['recipe.json'],source_index_sha256=fixed['source-index.json'],allowed_builds=['release','debug'],root_review_sha256=sha(HERE/'source-review001.json'),independent_review_sha256=sha(REVIEW/'receipt.json'),scope='single fixed local private Release001 and Debug001 helper/source gate only; no production/native/BH/operator/spectrum/evolution')
write(HERE/'authorization.json',auth)
write(HERE/'release.json',dict(owner=str(OWNER),authorization_sha256=sha(HERE/'authorization.json'),pins=pins))
print(json.dumps(dict(passed=True,pins=len(pins),authorization_sha256=sha(HERE/'authorization.json'))))
