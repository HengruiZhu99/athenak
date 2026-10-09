from pathlib import Path
import hashlib,json,subprocess
ROOT=Path('/Users/hz0693/research/hyperboloidal');HERE=Path(__file__).resolve().parent
OWNER=ROOT/'build-layer-research/boundary/inner-field-difference-arithmetic-supplement-v2-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/inner-field-difference-arithmetic-supplement-v2-full-independent-review-20261009'
NARROW=ROOT/'build-layer-research/continuum/inner-field-difference-supplement-v2-independent-admission-review-20261009'
V1=ROOT/'build-layer-research/boundary/inner-field-difference-arithmetic-supplement-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as s:s.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
indexhash='e7b31a31bef5ca4dd1745f9c68c0ba41d0c24718b38365016ae20f7e328d851e'
recipehash='bab172c1aec927c29cec9e423e645daad1ab69c45e190b386a914725896506ab'
if sha(OWNER/'source-index.json')!=indexhash or sha(OWNER/'recipe.json')!=recipehash:raise RuntimeError('source/recipe differs')
if sha(REVIEW/'index.json')!='237a7be9d2f01b07b3956d3732b03b3baf5bd38c2a1ce3fe33e73aad68660186' or sha(REVIEW/'receipt.json')!='271891854ec3a822ca8b2ae8d20505ca6fa7ffe66b5715afc4c0349782e9a18c':raise RuntimeError('full review differs')
if sha(NARROW/'receipt.json')!='bbcd4ab33ac350bb0a6672878d8e6131baa18ef370453184e4b3192269b53638':raise RuntimeError('narrow review differs')
for name in ['probe.cpp','oracle.py']:
 if (OWNER/name).read_bytes()!=(V1/name).read_bytes():raise RuntimeError('fixed original supplement science changed '+name)
r=load(OWNER/'recipe.json');index=load(OWNER/'source-index.json')
if r['expected_counts']!={'near-bound':108,'negative-old-near':3,'total':129,'witness':18} or r['actual_main003_overall_passed'] is not False or r['original_v1_ineligible_unchanged'] is not True:raise RuntimeError('fixed scope differs')
pins={x['path']:x['sha256'] for x in load(OWNER/'input-pins.json')}
for row in index['files']:pins[row['path']]=row['sha256']
pins[str(OWNER/'source-index.json')]=indexhash
for folder in [REVIEW,NARROW]:
 for p in folder.rglob('*'):
  if p.is_file():pins[str(p)]=sha(p)
for key in ['actual_main003_failed_receipt','actual_main003_failed_oracle','saved_association_summary','saved_association_receipt','saved_outer_identity_summary','saved_outer_identity_receipt']:
 row=r[key];pins[row['path']]=row['sha256']
pins[str(HERE/'prepare_release001.py')]=sha(__file__)
for p,d in pins.items():
 if sha(p)!=d:raise RuntimeError('protected input differs '+p)
main=load(r['actual_main003_failed_receipt']['path']);identity=load(r['saved_outer_identity_summary']['path'])
if not(main['passed'] is False and main['completed'] is False and main['returncode']==1 and main['source_inputs_unchanged'] is True and identity['passed_saved_identity_readback'] is True and identity['source003_original_failed'] is True and identity['all_failures_W_exactly_one'] is True and identity['failures_checked']==1872):raise RuntimeError('actualFAIL/W1 evidence differs')
if subprocess.run(['git','diff','--quiet','27c19d20696ea6dd4704032c51dfd026218f64f2','--','src','CMakeLists.txt'],cwd=ROOT).returncode:raise RuntimeError('production changed')
for name in r['attempt_names'].values():
 if (OWNER/'attempts'/name).exists():raise RuntimeError('one-shot output exists')
write(HERE/'source-review001.json',dict(passed=True,root_full_source_and_math_review=True,verified_unique_pins=len(pins),independent_full_review_sha256=sha(REVIEW/'receipt.json'),independent_narrow_review_sha256=sha(NARROW/'receipt.json'),source003_overall_FAIL_preserved=True,original_v1_ineligibility_preserved=True,checks=['three exact power normal targets and complete relative field-dual product rules','zero primal/nonzero gradient duals retained','all six boundary ratios and three reference powers finite normal product range','Fraction value/dual/branch counters plus exact129 unique labels','unchanged source002 old-near expressions yield three expected lost-normal zero controls','exact actual failed main and complete saved W1 qualification mandatory before compile','fixed compiler dependency closure, isolated unoptimized bytecode-off oracle and thread guards'],limits=['named field-difference units only; no gauge/outer/full22/operator/evolution acceptance','no dV or floating C1 or arbitrary-state accuracy claim'],no_scientific_import_compile_or_query_by_preparation=True))
write(HERE/'authorization.json',dict(field_difference_units_execution_admitted=True,recipe_sha256=recipehash,source_index_sha256=indexhash,allowed_builds=['release','debug'],root_review_sha256=sha(HERE/'source-review001.json'),independent_review_sha256=sha(REVIEW/'receipt.json'),scope='single local129 arithmetic-unit Release001 then conditional Debug001 only; main003 remainsFAILED'))
write(HERE/'release.json',dict(owner=str(OWNER),authorization_sha256=sha(HERE/'authorization.json'),pins=pins))
source=(ROOT/'build-layer-research/inner-joint-nonlinear-source003-root-release-20261009/launch.py').read_text()
if source.count("receipt.get('source_inputs_unchanged')")!=1:raise RuntimeError('launcher adaptation site differs')
with (HERE/'launch.py').open('x') as s:s.write(source.replace("receipt.get('source_inputs_unchanged')","receipt.get('inputs_unchanged')"))
print(json.dumps(dict(passed=True,pins=len(pins),authorization_sha256=sha(HERE/'authorization.json'))))
