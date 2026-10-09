from pathlib import Path
from fractions import Fraction
import hashlib,json,shutil
base=Path('/Users/hz0693/research/hyperboloidal');out=base/'build-layer-research/manufactured-angular-Gaussian-screen-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=out/'screen.py';s=p.read_text();old='    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)';new="    out=args.output.resolve()\n    if out.parent!=(HERE/'attempts').resolve():raise ValueError('fresh direct attempt child required')\n    out.mkdir(parents=True,exist_ok=False)"
if s.count(old)!=1:raise RuntimeError('source edit count')
p.write_text(s.replace(old,new))
old=json.loads((base/'build-layer-research/continuum/native-angular-pulse-flat-derivatives-compact-full-held-20261009/derivative-recipe.json').read_text())
pins=dict(old['mpmath_python_pins']);pins[old['python_runtime_path']]=old['python_runtime_sha256']
context=['build-layer-research/manufactured-angular-seed-admissibility-plan-held-20261009/index.json','build-layer-research/manufactured-angular-seed-admissibility-plan-held-20261009/PLAN.md','build-layer-research/manufactured-angular-seed-admissibility-plan-held-20261009/context.json','build-layer-research/continuum/manufactured-angular-time-wave-pencil-20261009/index.json','build-layer-research/continuum/manufactured-angular-time-wave-pencil-20261009/DERIVATION.md','build-layer-research/continuum/manufactured-angular-time-wave-independent-review-20261009/index.json','build-layer-research/continuum/manufactured-angular-time-wave-independent-review-20261009/receipt.json']
for rel in context:pins[str(base/rel)]=sha(base/rel)
recipe=dict(status='HELD source-only finite physical-event Gaussian screen',sigma=['7/20','1/2'],epsilon=['0','1/4','1/2','3/4'],a='1/2',precision_and_terms=[[80,40],[110,60]],precision_tolerance='1e-60',identity_tolerance='1e-65',radius_over_sigma=['0','1/1000000','1/10000','1/100','1/8','1/4','1/2','3/4','1','3/2','2','3','4','6','8','12','16','32','64','128','256','1024','1000000'],time_over_sigma=[str(Fraction(k,8)) for k in range(129)],retarded_time_over_sigma=[str(Fraction(k,4)) for k in range(-32,33)],pins=pins,python_runtime_path=old['python_runtime_path'],scope='Exact angular minimum at finitely many physical (T,R) events for pure-CMC height; native T depends on angle and is not solved. Positivity is a screen, never a global or native acceptance.',all_samples_must_be_preserved_including_negative=True,positive_D_required_for_scientific_identity_gate=False,no_eigen_kernel_inverse_or_evolution=True,environment={'PYTHONDONTWRITEBYTECODE':'1','PYTHONOPTIMIZE':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'})
events=sum(len({Fraction(t) for t in recipe['time_over_sigma']}|{Fraction(r)+Fraction(u) for u in recipe['retarded_time_over_sigma'] if Fraction(r)+Fraction(u)>=0}) for r in recipe['radius_over_sigma'])
recipe['anticipated_records_per_precision']=events*len(recipe['sigma'])*len(recipe['epsilon'])
(out/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
(out/'PLAN.md').write_text('''# Held finite Gaussian physical-event screen

This fresh source consumes the general-f pencil and root Gaussian seed plan.
It evaluates only C,C_T,C_R and the exact angular minimum of D at predeclared
physical (T,R) events for the pure CMC endpoint h=R/sqrt(R^2+a^2).
No actual native-time inverse, complete state jet, kernel source, operator,
continuum or evolution acceptance is inferred. All samples, including negative
D, are retained. A negative physical-event sample requires a separate native
inverse-domain check before calling it a future native slicing failure.

For R<=sigma/8, use the convergent entire Gaussian origin series
C=-sum_j 8(j+2)(j+1)f^(2j+5) R^(2j)/(2j+5)! and its analytic T/R
derivatives. The exact origin branch keeps C and C_T nonzero and sets C_R=0.
Else use independently displayed advanced/retarded C, C_T and C_R expressions.
Gaussian derivatives use the probabilists Hermite recurrence. The two fixed
runs use80digits/40terms and110digits/60terms; every matched D, D/referenceD
and J requires scaled difference<=1e-60. This is a precision/series comparison,
not an interval truncation proof.

The exact angular quadratic minimum uses s=1 and p in[-1/2,1/2]. Both endpoints
and any convex interior vertex are considered. At the selected p, a real
Cartesian sphere point and original vector-gradient formula independently
reconstruct D; scaled residual must be<=1e-65. The reference d0 is evaluated
as a^2/(R^2+a^2), never by subtracting nearly equal terms. Both signs and minima
are recorded. Positivity is reported separately from these identity gates.

Only metadata/source creation has occurred. Exact root authorization and an
independent source/math review are required before mpmath import or evaluation.
No permission for later inverse/jet/native steps is inferred.
''')
shutil.copyfile(Path(__file__),out/'prepare_metadata001.py')
files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
(out/'source-index.json').write_text(json.dumps(dict(source_only=True,execution_admitted=False,files=files),indent=2)+'\n')
print(json.dumps(dict(index_sha256=sha(out/'source-index.json'),source_sha256=sha(out/'screen.py'),recipe_sha256=sha(out/'recipe.json'),records_per_precision=recipe['anticipated_records_per_precision'],pins=len(pins))))
