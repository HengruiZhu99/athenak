from pathlib import Path
import hashlib,json,subprocess
ROOT=Path('/Users/hz0693/research/hyperboloidal');HERE=Path(__file__).resolve().parent
OWNER=ROOT/'build-layer-research/boundary/reference-wave-map-outer-arithmetic-source001-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/reference-wave-map-outer-arithmetic-independent-review-20261009'
OLD=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source003-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as s:s.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
fixed={'source-index.json':'9da21a96d243441c95f276d1c076d3a4e561fc20921858be33f911cb3464f27e','recipe.json':'0fd8f405f4c4d4343a3826f69247179c1236e672d5e81c20391093a30006d7c1','inputs/reference_wave_map.hpp':'f7a43a53481d3147c4394628de55651f8605fb6c0a8758b6b0a12116a5f9cfa8'}
for name,d in fixed.items():
 if sha(OWNER/name)!=d:raise RuntimeError('fixed source differs '+name)
if sha(REVIEW/'index.json')!='d3bae95086feedf0e2010af7705ecae9d3e1cef6f44f34ee4e1343ef5438e242' or sha(REVIEW/'receipt.json')!='1c86117c1a60640c96adf5bf7d7e9a8a442eb39ef269c89d5872c264831a427c':raise RuntimeError('review differs')
review=load(REVIEW/'receipt.json')
if not(review['source_math_passed'] and review['admission_passed_conditional_fresh_root_authorization'] and review['source_inputs_unchanged'] and not review['candidate_scientific_execution'] and not review['blocking_findings']):raise RuntimeError('independent review not admitted')
r=load(OWNER/'recipe.json');old=load(OLD/'recipe.json');index=load(OWNER/'source-index.json')
for key in ['expected_record_counts','thresholds','precision','modes','raw22_order','gauge_raw22_indices','geometry_raw22_indices','high_contrast_families']:
 if r[key]!=old[key]:raise RuntimeError('fixed science settings changed '+key)
if r['new_outer_contract']!={'far':'same real equations/new arithmetic, old baseline recorded; unchanged MP thresholds','near':'all8 split parts exact legacy values+duals at W1'}:raise RuntimeError('outer contract differs')
failure=OLD/'attempts/Release001/receipt.json'
if sha(failure)!='0f207c35ef15a0b765930677c5b1f220de9d5fd6ae05f04f32241fa9e58267c0':raise RuntimeError('source003 failure differs')
prior=load(failure)
if prior['passed'] or prior['completed'] or prior['returncode']!=1 or not prior['source_inputs_unchanged']:raise RuntimeError('source003 failure status differs')
pins={x['path']:x['sha256'] for x in load(r['input_pins'])}
for row in index['files']:pins[row['path']]=row['sha256']
pins[str(OWNER/'source-index.json')]=fixed['source-index.json']
for p in REVIEW.rglob('*'):
 if p.is_file():pins[str(p)]=sha(p)
pins[str(HERE/'prepare_release001.py')]=sha(__file__)
for p,d in pins.items():
 if sha(p)!=d:raise RuntimeError('input differs '+p)
principal=load(r['principal_receipt'])
if not(principal['completed'] and principal['passed'] and principal['returncode']==0 and principal['inputs_unchanged']):raise RuntimeError('actual principal gate prerequisite failed')
production=subprocess.run(['git','diff','--quiet','27c19d20696ea6dd4704032c51dfd026218f64f2','--','src','CMakeLists.txt'],cwd=ROOT,check=False)
if production.returncode!=0:raise RuntimeError('production baseline changed')
for name in r['attempt_names'].values():
 if (OWNER/'attempts'/name).exists():raise RuntimeError('single-use output exists')
root_review=dict(passed=True,verified_unique_pins=len(pins),root_full_source_and_math_review=True,independent_review_sha256=sha(REVIEW/'receipt.json'),production_identical_to='27c19d20696ea6dd4704032c51dfd026218f64f2',source003_overall_FAIL_preserved=True,original_15740_registry_and_MP_thresholds_unchanged=True,
 checks=['complete far dV/dL and physical-P RWM groups algebraically equal legacy equations','full nonflat reference connection and single pole assembly retained','source003 arithmetic extraction and inner Coefficient/Gauge suffix identical','joint positive primal near branch retains legacy value and field-dual bits at W1','far W1 old baseline retained as metadata; unchanged MP split/RHS thresholds mandatory','quoted include closure selects new wrapper and exact legacy header','zero primal retains nonzero registered first field-dual product terms','isolated unoptimized bytecode-off oracle after pins/environment/dependency guards'],
 limits=['finite local private point families only; no production/native/BH/operator/spectrum/evolution admission','far high-contrast field-dual supplement remains separate and unexecuted','no arbitrary metric/gradient contrast or universal well-conditioned summation certificate','no floating branch C1, regular scri or nonlinear stability claim'],no_scientific_import_compile_or_query_by_preparation=True)
write(HERE/'source-review001.json',root_review)
auth=dict(outer_arithmetic_local_execution_admitted=True,recipe_sha256=fixed['recipe.json'],source_index_sha256=fixed['source-index.json'],allowed_builds=['release','debug'],root_review_sha256=sha(HERE/'source-review001.json'),independent_review_sha256=sha(REVIEW/'receipt.json'),scope='single fixed local private Release001 then conditional Debug001 new outer arithmetic point gates only; old source003 FAIL remains FAIL')
write(HERE/'authorization.json',auth)
write(HERE/'release.json',dict(owner=str(OWNER),authorization_sha256=sha(HERE/'authorization.json'),pins=pins))
print(json.dumps(dict(passed=True,pins=len(pins),authorization_sha256=sha(HERE/'authorization.json'))))
