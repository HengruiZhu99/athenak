"""Root source/math review, with a pre-execution C++ driver admission finding."""
from pathlib import Path
import ast,hashlib,json,sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SRC=REPO/'build-layer-research/continuum/exact-signed-product-sum-source001-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
assert sha(SRC/'source-index.json')=='0f21b208093a5fe042e7e8f02b300380ac0a88694bf3afee5bbee0b214e72460'
assert sha(SRC/'recipe.json')=='ab815f1e83bcd4d6e7fae2224a474337bdfbead3d98c83564d1c2ecf86114a26'
pins={r['path']:r['sha256'] for r in load(SRC/'source-index.json')['files']+load(SRC/'external-pins.json')}
pins[str(SRC/'source-index.json')]=sha(SRC/'source-index.json')
for p,h in pins.items():assert sha(p)==h,p
for name in ('fraction_oracle.py','run_gate.py'):ast.parse((SRC/name).read_text())
r=load(SRC/'recipe.json');registry=load(SRC/'registry.json')
assert r['fixed_cases']==registry['case_count']==70 and registry['scalar_count']==44 and registry['dual_count']==26
assert len(registry['cases'])==70 and len({x['id'] for x in registry['cases']})==70
assert Path(r['compiler']['path']).name=='clang'
review={
 'passed_source_review':False,'source_math_algorithm_review_passed':True,'execution_released':False,
 'reviewed_source_index_sha256':sha(SRC/'source-index.json'),'recipe_sha256':sha(SRC/'recipe.json'),
 'protected_paths':len(pins),'inputs_unchanged':True,'compiler_or_numerical_targets_executed':False,
 'finding':'Recipe invokes resolved clang rather than the clang++ driver name while linking C++ iostream/string/vector; no explicit C++ standard library link flag. Fresh source002 must preserve literal clang++ invocation and separately pin resolved binary contents. No actual compile failure has occurred.',
 'full_root_reads':['PLAN.md','IMPLEMENTATION.md','PREPARATION.md','signed_products.hpp','probe.cpp','fraction_oracle.py','run_gate.py','recipe.json','registry.json control definitions/IDs/statuses and cases metadata'],
 'reviewed_math':[
  'Finite binary64 atom sign/significand/exponent decoding is exact; four53-bit integer significands fit212bits in four64-bit limbs; unsigned128 word-times-significand plus carry is below2^117.',
  'Minimum admitted product exponent -4297 exceeds BASE_E=-4352; at most128 monomials of magnitude below2^4096 give separately signed accumulators below2^4103 within136words.',
  'AddWord propagation and shifted four-limb AddProduct are exact, with all shifts defined; Subtract uses Wide rhs for the b[i]+borrow==2^64 edge.',
  'Maxfinite exact comparison precedes nearest-even encoding; normal retain53bits and subnormal2^-1074 grid use guard/sticky/odd correctly; normal carry/subnormal-to-normal carry and nonzero-negative rounded zero handled.',
  'AnyBelow and Quotient indices remain within136words after maxfinite guard; normal trim is at least3278, subnormal trim exactly3278.',
  'All consumed primal/tangent atoms validated before zero shortcuts; unused atoms excluded by contract; zeroarity constants and nullptr/count validation are distinct.',
  'EvaluateDual generates every factor replacement term, including zero primals, repeated factors and zero tangents, with at most128 terms; no division by primal or seed recognition.',
  'Independent Fraction formal polynomial recurrence and CPython rational-to-float conversion differ structurally from integer candidate; explicit hand bit/status controls add fixed tie/overflow expectations.',
  'Fixed44scalar+26dual controls cover cancellation/range/ties/subnormal/signedzero/invalid/independentoverflow/128term edges; exact result/status/echo/audit gates, no tolerance relaxation.',
  'Unit runner binds source/recipe/review/runtime/header universe, requires fresh destination and zero stderr, checks compiler dependencies, Release/ASanUB oracle reports and byte-identical stdout.'
 ],
 'scope':'Submitted binary64 product sums only; cannot recover rounded inverse inputs; no RWM replacement, original63-failure upgrade, native PDE or evolution admission.',
 'read_order':'Root text reads preceded this receipt; supplied exact source-index identity was subsequently rehashed in full. No false pre-pinning claim.'
}
write(HERE/'source001-pins.json',pins);write(HERE/'source001-review.json',review)
print(json.dumps({'source_math_passed':True,'execution_released':False,'source_admission_passed':False,'pins':len(pins),'review_sha256':sha(HERE/'source001-review.json')}))
