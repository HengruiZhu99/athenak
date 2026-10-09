from pathlib import Path
import hashlib,json,ast
ROOT=Path('/Users/hz0693/research/hyperboloidal');HERE=Path(__file__).resolve().parent
OWNER=ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-native-inverse-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
fixed={'source-index.json':'27be52a7a76657a6bf7544f21d2a7357b19f4e101950385314a7171c9ddf64a2','recipe.json':'99acf179d6d11dd81d2908ed5b7e23b3cd73519522cce087e03a82e4e090ebc1','screen.py':'3224d7aec6e6c53858c2f748548c1f246b7eaa100aaf9062a69782be976c0c04','launch_once.py':'12adbbcef07ffc123d591029ffea857728c44a2004b1ddea1be71a0b7751de89','values_context.py':'89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7'}
for n,d in fixed.items():
 if sha(OWNER/n)!=d:raise RuntimeError('fixed source differs '+n)
r=load(OWNER/'recipe.json');idx=load(OWNER/'source-index.json');pins=dict(r['pins'])
for row in idx['files']:pins[row['path']]=row['sha256']
pins[str(OWNER/'source-index.json')]=fixed['source-index.json']
old=ROOT/'build-layer-research/manufactured-angular-Gaussian-screen-v2-held-20261009/screen.py'
new=OWNER/'gaussian_radial.py'
a,b=old.read_text(),new.read_text()
for name in ['derivatives','radial']:
 na=next(x for x in ast.walk(ast.parse(a)) if isinstance(x,ast.FunctionDef) and x.name==name)
 nb=next(x for x in ast.walk(ast.parse(b)) if isinstance(x,ast.FunctionDef) and x.name==name)
 if ast.get_source_segment(a,na)!=ast.get_source_segment(b,nb):raise RuntimeError('Gaussian source definition changed '+name)
previous=ROOT/'build-layer-research/manufactured-angular-Gaussian-screen-v2-held-20261009/attempts/screen001/receipt.json'
if sha(previous)!='3b46677afd4dce07e38c18ddf3d41e0497e9e44a88759b2929b1ee30cee12979':raise RuntimeError('previous physical screen receipt differs')
pins[str(previous)]=sha(previous)
for name,d in [('manufactured-angular-time-wave-independent-review-20261009/index.json','98b6732fc6081cb0d6de5ad299309d1138c07c1b55278e0e59e8142efda7e5f6')]:
 q=ROOT/'build-layer-research/continuum'/name
 if sha(q)!=d:raise RuntimeError('general-f math review differs')
 pins[str(q)]=d
if len(r['native_radii'])!=21 or len(r['native_times'])!=9 or len(r['p_values'])!=9 or r['records_per_level']!=13608 or r['total_records']!=54432:raise RuntimeError('event grid changed')
if r['newton_iterations']!=16 or r['bisection_iterations']!=512 or r['root_absolute_tolerance']!='1e-50' or r['root_width_tolerance']!='1e-55' or r['comparison_tolerance']!='1e-30':raise RuntimeError('threshold changed')
if r['python_runtime_sha256']!=sha(r['python_runtime_path']):raise RuntimeError('actual interpreter changed')
for p,d in pins.items():
 if sha(p)!=d:raise RuntimeError('protected input changed '+p)
for n in ['attempt001','outer-invocation001']:
 if (OWNER/n).exists():raise RuntimeError('single-use destination exists')
review=dict(passed_source_math_review=True,root_read_complete_screen_wrapper_plan_gaussian_extraction=True,verified_unique_pins=len(pins),source_index_sha256=fixed['source-index.json'],no_scientific_import_or_execution_in_preparation=True,
 checks=['Gaussian definitions byte-exact to actual accepted physical screen','full fixed 21x9x9x8 grid at four separate precision/height levels','global J lower bound and exact even-seed inverse domain','core global F bracket and outer bounded c_ret bracket retain advanced terms','16 safeguarded Newton plus512 bisection with explicit numerical endpoint signs/residual/width','outer Phi,Aplus,Aminus,Delta identities independently derived from original map','direct MP vector-gradient normalized D and original-map residual cross-checks','conformal ADM values derived by radial pullback; only positive D admits real ADM branch','separate precision and height comparisons; negative D rows retained','actual isolated interpreter and MP/helper origin guards before science'],
 limits=['finite sampled inverse/value consistency only; no full native domain positivity','computed signs and MP quadrature are not rigorous interval certificates','ADM values only; no curvature/connection/time jets or kernel/PDE/evolution','manufactured physical RWM solution does not solve modified-BM inner helper','no original native-pulse caustic verdict, scri closure, BH or stability acceptance'])
write(HERE/'source-review001.json',review)
auth=dict(native_Gaussian_inverse_screen_authorized=True,source_index_sha256=fixed['source-index.json'],recipe_sha256=fixed['recipe.json'],screen_source_sha256=fixed['screen.py'],outer_source_sha256=fixed['launch_once.py'],root_source_math_review_sha256=sha(HERE/'source-review001.json'),scope='one fixed finite native-time Gaussian inverse/value consistency screen; positivity and PDE stability separate')
write(HERE/'authorization.json',auth)
write(HERE/'release.json',dict(owner=str(OWNER),pins=pins,authorization_sha256=sha(HERE/'authorization.json')))
print(json.dumps(dict(passed=True,pins=len(pins),authorization_sha256=sha(HERE/'authorization.json'))))
