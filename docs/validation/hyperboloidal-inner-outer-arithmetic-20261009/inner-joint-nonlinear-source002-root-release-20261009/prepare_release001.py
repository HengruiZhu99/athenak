from pathlib import Path
import hashlib,json,subprocess
ROOT=Path('/Users/hz0693/research/hyperboloidal');HERE=Path(__file__).resolve().parent
OWNER=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source002-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/inner-joint-nonlinear-helper-source002-independent-review-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as s:s.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
fixed={'source-index.json':'a6dfac99b3b9296629701bfd02f90ad79c2b36f47e95c747d69c897d694ade05','recipe.json':'2c1dde08c070f0a7bcde6d66123bc11378253c0365002393a9968b00529292c9','inner_gauge.hpp':'b98e1bd55c21c8824312a8b4e5d2b150c7f87f67ace43953c56b46a890d2919b','probe.cpp':'3d7a83d6b10881327a93c72fb947f4f8e479c715c8c1d3f8484fb30228f46c09','oracle.py':'15fd453129045b45348228f6dce66bd2eba2f05daacadae448ffc0f301922160','run_once.py':'8eea82600dcbebda98a8e8e95cbd3427ffdc30b74e40c8084207d0595e666a92'}
for name,d in fixed.items():
 if sha(OWNER/name)!=d:raise RuntimeError('fixed source differs '+name)
if sha(REVIEW/'index.json')!='18563ea4eea29878153cc7af57b89afb21cc40f65f7c306fcded534c24b398a8' or sha(REVIEW/'receipt.json')!='68eeb66dcd0193d882da8ca72ad963f8a8255581ccb1b5a98614ec4783a45a25':raise RuntimeError('review differs')
review=load(REVIEW/'receipt.json')
if not(review['status'].startswith('PASS') and review['inputs_unchanged'] and review['no_candidate_import_compile_syntax_CAS_numerics_array_query_operator_evolution']):raise RuntimeError('independent review not admitted')
PREVIOUS=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source001-held-20261009'
PRIOR_ROOT=ROOT/'build-layer-research/inner-joint-nonlinear-source001-root-release-20261009'
a=(PREVIOUS/'probe.cpp').read_bytes();b=(OWNER/'probe.cpp').read_bytes()
before=b'const auto qp=Parts(q),fp=Fields(f);';after=b'const auto qp=Parts(q);const auto fp=Fields(f);'
if a.count(before)!=1 or a.replace(before,after)!=b:raise RuntimeError('nonminimal scientific probe change')
for name in ['inner_gauge.hpp','oracle.py','run_once.py']:
 if (PREVIOUS/name).read_bytes()!=(OWNER/name).read_bytes():raise RuntimeError('scientific source changed '+name)
failure=PREVIOUS/'attempts/Release001/receipt.json'
if sha(failure)!='e965eda26ff716e62d30764a210953550572eca452b06efd542e045502eaead6' or load(failure)['completed'] or load(failure)['returncode']!=1:raise RuntimeError('prior compile failure differs')
if sha(PREVIOUS/'attempts/Release001/compile.stderr')!='27e417a29bda31d849a6f189bbbe7563b4ff3dd6984496d9a8661ea274919fdd':raise RuntimeError('compiler stderr differs')
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
root_review=dict(passed=True,verified_unique_pins=len(pins),root_full_source_and_math_review=True,source002_exact_single_declaration_split=True,prior_source001_compile_failure_preserved=True,prior_root_review_sha256=sha(PRIOR_ROOT/'source-review001.json'),independent_review_sha256=sha(REVIEW/'receipt.json'),production_identical_to='27c19d20696ea6dd4704032c51dfd026218f64f2',
 checks=['complete reference physical connection and grouped literal algebra','P stored with independent Theta, no K/Theta substitution','bounded k variable and relative field-dual derivatives','W1 exact baseline return before inner arithmetic','independent MP split/core witness target; huge A cancellation not claimed at insufficient precision','actual22 geometric rows identity plus independent gauge dual/19 principal coefficients','all three FD levels saved; final-level and convergence/floor gates mandatory','source/input guards and unoptimized isolated oracle before import'],
 limits=['declared finite local families only; no native or puncture/BH/global stability','dc/dal grouping can lose tiny relative gradients far from reference; accuracy not certified for all positive inputs','extreme coefficient AD checked for relative seeds only; finite q.valid does not certify arbitrary seed accuracy'],no_scientific_import_compile_or_query_by_preparation=True)
write(HERE/'source-review001.json',root_review)
auth=dict(local_nonlinear_helper_execution_admitted=True,recipe_sha256=fixed['recipe.json'],source_index_sha256=fixed['source-index.json'],allowed_builds=['release','debug'],root_review_sha256=sha(HERE/'source-review001.json'),independent_review_sha256=sha(REVIEW/'receipt.json'),scope='single fixed local private Release001 and Debug001 helper/source gate only; no production/native/BH/operator/spectrum/evolution')
write(HERE/'authorization.json',auth)
write(HERE/'release.json',dict(owner=str(OWNER),authorization_sha256=sha(HERE/'authorization.json'),pins=pins))
print(json.dumps(dict(passed=True,pins=len(pins),authorization_sha256=sha(HERE/'authorization.json'))))
