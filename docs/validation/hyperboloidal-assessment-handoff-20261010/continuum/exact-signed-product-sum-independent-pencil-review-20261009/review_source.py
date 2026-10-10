"""Source-only stdlib review: no oracle, arithmetic unit, import or compile."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OWNER=ROOT/'build-layer-research/continuum/exact-signed-product-sum-source002-held-20261009'
OLD=OWNER.with_name('exact-signed-product-sum-source001-held-20261009')
EXPECTED='36c72f687bad50fa61c3f1ac684cf5b9b28cfa99348e71e627ea60bad5489a09'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def save(path,v):Path(path).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def require(value,message):
 if not value:raise RuntimeError(message)
start=time.monotonic()
require(not (HERE/'receipt.json').exists(),'one-shot independent review')
require(sha(OWNER/'source-index.json')==EXPECTED,'exact source002 index')
index=load(OWNER/'source-index.json');external=load(OWNER/'external-pins.json')
pins={}
for p in external+index['files']:
 require(p['path'] not in pins or pins[p['path']]==p['sha256'],'pin conflict')
 pins[p['path']]=p['sha256']
pins[str(OWNER/'source-index.json')]=EXPECTED
before={name:sha(name) for name in pins};require(before==pins,'before pin equality')
save(HERE/'pins-before.json',before)
unchanged=[]
for name in ('signed_products.hpp','probe.cpp','fraction_oracle.py','cases.txt','registry.json','PLAN.md'):
 require((OWNER/name).read_bytes()==(OLD/name).read_bytes(),'science source001/source002 equality '+name)
 unchanged.append({'file':name,'sha256':sha(OWNER/name),'byte_equal_source001':True})
registry=load(OWNER/'registry.json');recipe=load(OWNER/'recipe.json')
require(len(registry['cases'])==70 and sum(x['mode']=='scalar' for x in registry['cases'])==44 and sum(x['mode']=='dual' for x in registry['cases'])==26,'fixed registry counts')
require(recipe['compiler']['path']=='/Library/Developer/CommandLineTools/usr/bin/clang++','literal C++ driver basename')
require(str(Path(recipe['compiler']['path']).resolve())==recipe['compiler']['resolved_path'],'driver target')
require('-fno-fast-math' in recipe['release_flags'] and '-fno-fast-math' in recipe['debug_flags'],'strict compiler arithmetic')
for p in OWNER.glob('*.py'):ast.parse(p.read_text())
header=(OWNER/'signed_products.hpp').read_text();runner=(OWNER/'run_gate.py').read_text();oracle=(OWNER/'fraction_oracle.py').read_text()
require('constexpr std::size_t kWords = 136;' in header and 'constexpr int kBaseExponent = -4352;' in header,'capacity constants')
require('if (shift != -1 && shift != 0)' in header and 'if (arity > kFactors)' in header,'bounded domain')
require('for (std::size_t differentiated = 0; differentiated < t.arity; ++differentiated)' in header,'complete polynomial dual expansion')
require('Compare(magnitude, MaxFinite()) > 0' in header,'strong exact-overflow domain')
require('guard && (sticky || (q & 1U) != 0)' in header,'ties-even guard/sticky')
require('Path(args.recipe).resolve() == P / \'recipe.json\'' in runner,'consumed recipe binding')
require("review.get('reviewed_source_index_sha256') == sha(P / 'source-index.json')" in runner,'review exact-index admission')
require("'-I', '-B'" in runner and 'sys.flags.optimize == 0' in runner,'isolated unoptimized Python route')
require('d, p = d * v + p * dv, p * v' in oracle and 'rounded = float(exact)' in oracle,'structurally independent Fraction oracle')
save(HERE/'source-checks.json',{'unchanged_scientific_sources':unchanged,'source002_literal_driver_and_target_binding':True,'fixed_case_counts':[70,44,26],'all_Python_AST_parsed_without_import':True,'capacity_and_rounding_proof':'independent handwritten source review in REVIEW.md','Fraction_targets_evaluated':False})
copy=HERE/'source-copies';copy.mkdir()
for row in index['files']:
 p=Path(row['path']);require(p.stat().st_size<=1048576 and p.suffix not in ('.npz','.npy','.jsonl'),'compact source capture')
 shutil.copyfile(p,copy/p.name)
shutil.copyfile(OWNER/'source-index.json',copy/'owner-source-index.json')
(HERE/'REVIEW.md').write_text('''# Independent exact signed-product-sum pencil/source review\n\nPASS for the bounded standalone source002 arithmetic proposal. No numerical module, Fraction target, C++ unit, compiler, saved array or gauge query was executed. Source002's literal clang++ invocation and separately pinned resolved clang target correct the preserved source001 admission risk. This review does not inspect any actual unit result or authorize adoption.\n\nFor a finite binary64 atom, the integer significand is below2^53 and its stored exponent lies between-1074 and971. A four-factor monomial has at most212 significand bits. The smallest exponent including a half coefficient is-4297, above BASE=-4352 by55 bits. At most128 same-sign monomials have magnitude below2^4103, while136 words cover the integer lattice from2^-4352 through the bit at2^4351. This leaves adequate high and low capacity. Word-times-significand plus carry fits unsigned128; word addition plus carry and the borrow subtraction preserve exact limb arithmetic. The public scalar limit remains32;128 is the internally generated complete dual limit only.\n\nAddProduct's split low/high word additions reconstruct the shifted exact product. Positive and negative accumulators remain separate until one exact comparison/subtraction. The guard and sticky routines act below the retained53-bit normal grid or the fixed2^-1074 subnormal grid. Increment on guard and (sticky or odd) is nearest/even. A normal carry renormalizes; a subnormal carry directly encodes the minimum normal. The exact comparison with maxfinite prevents accepted overflow. It deliberately rejects all exact sums greater than maxfinite, even ones that could round back to maxfinite. This stronger domain is explicit in both source and oracle.\n\nCanonical exact cancellation returns+0. A negative nonzero exact sum that rounds to zero retains-0. The local half-grid exponent is retained as an integer, so its bound is meaningful below representable binary64. It does not certify error already present in submitted factors or error in a surrounding source expression. All consumed atoms are validated before zero-product shortcuts; unused slots are outside the stated polynomial domain.\n\nThe dual expansion replaces each factor in turn by its tangent and retains every generated term, including zero-primal/nonzero-tangent cases and zero derivative terms. It has no division by a primal or seed-specific recognition. The Fraction oracle independently uses a formal-polynomial recurrence and CPython rational conversion, then separately checks expanded counters. Its70 fixed controls include ordinary/subnormal ties, signed zero, binade carry, cancellation after extreme intermediate products, invalid domains, complete128 derivatives and independent primal/tangent overflow. This is finite evidence for the reviewed algorithm, not exhaustive enumeration.\n\nThe one-shot runner binds the exact consumed recipe, source index, root release, passed same-source review, runtime/environment and compiler invocation/target. Scientific sources are byte-identical to source001. Release and ASan/UBSan probe output equality, exact oracle checks and compiler dependency closure remain requirements of a separately released actual unit stage. No gauge includes this header; the far-dual source failures and inverse/auxiliary errors remain unresolved.\n''')
after={name:sha(name) for name in pins};require(after==before,'after pin equality');save(HERE/'pins-after.json',after)
save(HERE/'receipt.json',{'passed_source_review':True,'passed':True,'reviewed_source_index_sha256':EXPECTED,'status':'PASS_SOURCE_ONLY_PENCIL_AND_ADMISSION_REVIEW','protected_unique_pins':len(pins),'fixed_cases':70,'scalar_cases':44,'dual_cases':26,'source_only':True,'candidate_imports':False,'Fraction_targets_evaluated':False,'compiler_or_query_calls':False,'actual_unit_outputs_inspected':False,'before_after_unchanged':True,'metadata_seconds':time.monotonic()-start})
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p!=HERE/'index.json')
save(HERE/'index.json',{'status':'immutable independent source-only pencil review','files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
print(json.dumps({'index':sha(HERE/'index.json'),'receipt':sha(HERE/'receipt.json'),'review':sha(HERE/'REVIEW.md'),'files':len(files),'pins':len(pins)}))
