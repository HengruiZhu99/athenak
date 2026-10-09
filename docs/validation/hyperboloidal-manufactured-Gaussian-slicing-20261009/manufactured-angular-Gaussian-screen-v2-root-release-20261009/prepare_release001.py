from pathlib import Path
import ast,hashlib,json
ROOT=Path('/Users/hz0693/research/hyperboloidal');HERE=Path(__file__).resolve().parent
OWNER=ROOT/'build-layer-research/manufactured-angular-Gaussian-screen-v2-held-20261009'
OLD=ROOT/'build-layer-research/manufactured-angular-Gaussian-screen-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-screen-v2-independent-guard-review-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as s:s.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
fixed={'source-index.json':'287f96ced076617c877d4bfbd7d5d63e484dd8b2d2eb61a53c96e1e2f60fbf7b','recipe.json':'5b22ca77da59d52357d77bc9e00b20d5aace8ecfaa1c9c78dbbc4c84a4df0561','screen.py':'6c9cae79f014d30a81717d05320c865ff406415afd2c0bf249b3571737636c8d'}
for name,d in fixed.items():
 if sha(OWNER/name)!=d:raise RuntimeError('source differs '+name)
if sha(REVIEW/'index.json')!='2fd8aa3ff31281b2773c220cb56fa4fdea96ace208ce024a92a09a8c288aaaa2' or sha(REVIEW/'receipt.json')!='ad646b879df82d33b0e56763fad3d606b64b3268f0accd462034cbcb569e231b':raise RuntimeError('independent review differs')
review=load(REVIEW/'receipt.json')
if not(review['source_only_review_passed'] and review['original_v1_and_v2_inputs_unchanged'] and review['complete_scientific_body_byte_equal'] and not review['scientific_execution']):raise RuntimeError('review prerequisite fails')
recipe=load(OWNER/'recipe.json');index=load(OWNER/'source-index.json');pins=dict(recipe['pins'])
for row in index['files']:pins[row['path']]=row['sha256']
pins[str(OWNER/'source-index.json')]=fixed['source-index.json']
for p in REVIEW.iterdir():
 if p.is_file():pins[str(p)]=sha(p)
for p,d in pins.items():
 if sha(p)!=d:raise RuntimeError('pin differs '+p)
def fn(p):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
if ast.dump(ast.Module(body=fn(OWNER/'screen.py').body[3:],type_ignores=[]),include_attributes=False)!=ast.dump(ast.Module(body=fn(OLD/'screen.py').body[1:],type_ignores=[]),include_attributes=False):raise RuntimeError('scientific body changed')
if recipe['anticipated_records_per_precision']!=28032:raise RuntimeError('fixed count differs')
if (OWNER/'attempts/screen001').exists():raise RuntimeError('single-use attempt already exists')
root_review=dict(passed=True,verified_unique_pins=len(pins),independent_review_sha256=sha(REVIEW/'receipt.json'),root_reviewed_full_source_and_general_compact_pencils=True,
 mathematical_checks=['Gaussian Hermite derivatives and analytic entire origin series C,CT,CR','full advanced/retarded C_R signs','sphere min at s=1 and endpoints/convex vertex','independent original vector-gradient D reconstruction','global J and Cauchy-height bound does not certify CMC native slicing','precision/series comparison remains finite-screen evidence only'],
 no_scientific_import_or_execution_by_preparation=True,scientific_body_unchanged=True,global_or_native_admission=False)
write(HERE/'source-review001.json',root_review)
auth=dict(finite_Gaussian_screen_authorized=True,recipe_sha256=fixed['recipe.json'],source_index_sha256=fixed['source-index.json'],screen_source_sha256=fixed['screen.py'],root_review_sha256=sha(HERE/'source-review001.json'),independent_review_sha256=sha(REVIEW/'receipt.json'),scope='one finite physical-event Gaussian CMC D/angular-identity screen only; no full-domain/native inverse/jet/PDE/evolution admission')
write(HERE/'authorization.json',auth)
write(HERE/'release.json',dict(owner=str(OWNER),authorization_sha256=sha(HERE/'authorization.json'),pins=pins))
print(json.dumps(dict(passed=True,pins=len(pins),authorization_sha256=sha(HERE/'authorization.json'))))
