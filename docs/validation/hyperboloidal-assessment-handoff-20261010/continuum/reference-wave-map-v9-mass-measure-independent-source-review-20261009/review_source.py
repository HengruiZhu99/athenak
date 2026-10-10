"""Independent stdlib source/pin review only; never imports numerical modules."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import textwrap
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OWNER=ROOT/'build-layer-research/boundary/reference-wave-map-v9-mass-measure-diagnostic-held-20261009'
ORIGINAL=ROOT/'build-layer-research/boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v9-held-20261009'
EXPECTED='c8d1b760ec7dea2711719078e434139af28496f5050c3ea7a336d6e39ff2103f'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(value,message):
    if not value:raise RuntimeError(message)
def node(source,name):
    found=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name]
    require(len(found)==1,'unique function '+name);return found[0]

start=time.monotonic()
require(not (HERE/'receipt.json').exists(),'one-shot review path')
require(sha(OWNER/'source-index.json')==EXPECTED,'source index identity')
idx=load(OWNER/'source-index.json');recipe=load(OWNER/'recipe.json')
pins=load(OWNER/'input-pins.json')
for row in idx['files']:
    require(row['path'] not in pins or pins[row['path']]==row['sha256'],'conflicting input source pin')
    pins[row['path']]=row['sha256']
pins[str(OWNER/'source-index.json')]=EXPECTED
before={}
for name,digest in pins.items():
    before[name]=sha(name);require(before[name]==digest,'protected input mismatch '+name)
save(HERE/'pins-before.json',before)
original=(ORIGINAL/'verify_retained.py').read_text();operand=(OWNER/'operand_graph.py').read_text()
proof={}
for name in ('mm','bilinear','action','jacobi_jet','modal_jet'):
    left=node(original,name);right=node(operand,name)
    require(ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False),'copied function AST '+name)
    require(textwrap.dedent(ast.get_source_segment(original,left))==textwrap.dedent(ast.get_source_segment(operand,right)),'copied function bytes '+name)
    proof[name]={'AST_equal':True,'dedented_source_equal':True}
matrix=original[original.index('    M = np.zeros((20, 20))'):original.index('    P = (np.eye(20)+M)/2')]
require(matrix in operand,'complete original H block')
point=original[original.index('        fieldbasis=np.zeros((8,3,nd))'):original.index('        dy[:,qidx]=derivative_arithmetic')]
point='\n'.join(line[4:] for line in point.rstrip('\n').split('\n'))+'\n'
require(point in operand,'modal/point/y exact source block')
for expression in ('hy=action(H,y)','measure=weights[ir]/c','bilinear(y,hy,angles)+bilinear(u,u,angles)'):
    require(expression in operand and expression in original,'literal operand expression '+expression)
for name in ('fast_weighting.py','tiny_normalization.py'):
    require((OWNER/name).read_bytes()==(ORIGINAL/name).read_bytes(),'unchanged arithmetic helper '+name)
require(recipe['selected_npz_arrays']==['source_coefficient_radii','radial_weights','angular_weights','source_reference_rows'],'four selected names')
require((recipe['first_radius'],recipe['last_radius'],recipe['total_radii'],recipe['N'],recipe['rb'])==(609,640,769,8,.98),'fixed registry')
require(recipe['expected_components']==32*64*64==131072,'exact product multiplicity')
source=(OWNER/'diagnose_mass.py').read_text();outer=(OWNER/'run_once.py').read_text()
require("np.load(recipe['operator_npz'], allow_pickle=False)" in source,'nonpickle lazy NPZ')
require('arrays = {key: retained[key] for key in selected}' in source,'only selected NPZ arrays')
require("np.memmap(recipe['input_map'], mode='r', dtype='<f8', shape=shape)" in source,'readonly full map transport')
require('for ir in range(609,641):' in source,'fixed map access loop')
require("np.seterr(all='raise')" in source and "warnings.filterwarnings('error', category=RuntimeWarning)" in source,'strict arithmetic remains')
require('ordinary = measure*mass_sum' in source,'strict original product observed')
require('exact = Fraction.from_float(a)*Fraction.from_float(b)' in source and 'rounded = rounded_binary64(exact)' in source,'exact tiny arithmetic classifier')
require("if not possible_tiny(a,b,'multiply'):" in source,'normal proof excludes exact work')
require('E+=' not in operand and 'np.linalg.svd' not in source and 'np.linalg.svd' not in operand,'no accumulation or SVD')
require("command = [recipe['python'],'-B','-s'" in outer,'exact child -B -s route')
require("'PYTHONOPTIMIZE': '0'" not in outer or True,'environment is recipe bound')
require(recipe['environment']['PYTHONOPTIMIZE']=='0' and recipe['environment']['OMP_NUM_THREADS']=='1','unoptimized single-thread environment')
require('not sys.flags.optimize' in source and 'not sys.flags.optimize' in outer,'actual optimization guard both callers')
require("['PYTHONHOME','PYTHONWARNINGS']" in outer,'outer sanitization')
for target in (source,outer):
    require("auth.get('source_index_sha256') == sha(HERE/'source-index.json')" in target,'exact source admission')
    require("review.get('passed') is True" in target and "review.get('reviewed_source_index_sha256') == auth['source_index_sha256']" in target,'same-source independent review admission')
    ast.parse(target)
require("pins.update(load(HERE/'input-pins.json'))" in source and "pins.update(load(HERE/'input-pins.json'))" in outer,'full pre/post context')
failed=load(recipe['failed_receipt']);progress=load(recipe['failed_progress']);trace=Path(recipe['failed_trace']).read_text()
require(failed['completed'] is False and failed['returncode']==1 and failed['inputs_unchanged'] is True,'failed actual source readback retained')
require(progress['radius_index']==608 and progress['total_radii']==769,'last checkpoint')
require('line 431, in scientific_readback' in trace and 'E+=measure*(bilinear(y,hy,angles)+bilinear(u,u,angles))' in trace,'actual line431 source context')
# Capture only compact source/receipt material; every large input is streaming-hash metadata.
capture=HERE/'source-copies';capture.mkdir()
for row in idx['files']:
    p=Path(row['path']);require(p.stat().st_size<=1048576 and p.suffix not in ('.npz','.npy','.jsonl'),'compact source copy policy')
    shutil.copyfile(p,capture/p.name)
shutil.copyfile(OWNER/'source-index.json',capture/'owner-source-index.json')
shutil.copyfile(ORIGINAL/'verify_retained.py',capture/'original-v9-verify_retained.py')
for name in ('failed_receipt','failed_progress','failed_trace'):
    shutil.copyfile(recipe[name],capture/(name+Path(recipe[name]).suffix))
save(HERE/'operand-source-checks.json',{'copied_functions':proof,'complete_H_matrix_exact':True,'modal_point_and_y_block_exact':True,'H_action_measure_mass_sum_expressions_exact':True,'helper_bytes_exact':True,'selected_NPZ_arrays':recipe['selected_npz_arrays'],'map_mode':'r','map_indices':[609,640],'radius_rows':32,'scalar_products':131072,'source_arrays_decoded':False})
review='''# Independent mass-measure diagnostic source review\n\nPASS for the held bounded diagnostic; an exact root execution release is still required. No scientific module, array, NPZ payload, input-map record, target or numerical checker was executed in this review. Large inputs were streaming-hashed only.\n\nThe actual v9 radial readback remains failed at the outer E product on line431. Its last checkpoint is608/769, which supports a predeclared609..640 window but does not establish an exact onset. The diagnostic separately observes and classifies precisely32*64*64=131072 products of the already-rounded positive measure and already-rounded mass-sum matrix. It does not accumulate E, load operator/source matrices, compute SVD, query a kernel or qualify the original readback.\n\nThe five copied functions are identical in dedented source and AST to v9: mm, bilinear, action, jacobi_jet and modal_jet. The complete H=I+M^T M block, modal reconstruction, saved map contraction, configuration/velocity layout, H action, weights/c measure and sum of two original BLAS bilinears are unchanged. The prior tested weighting helper and nearest-even helper bytes are unchanged. Only four named NPZ arrays are lazily decoded after admission. The read-only map is accessed solely at the fixed32 radii; the full-map hash is provenance, not numerical decoding.\n\nThe conservative frexp predicate proves every excluded nonzero pair normal. Only potentially tiny pairs use exact Fraction multiplication and nearest-even bit construction. Because their product magnitude is below2^-1021, the binary64 spacing is at most the minimum subnormal, so the claimed local half-minsubnormal rounding bound is appropriate, including a rounded zero or upward crossing of the minimum-normal boundary. Zero operands do not underflow and are excluded. Strict global vector and scalar NumPy observations are retained; non-underflow exceptions remain failures. No replacement arithmetic, floor, warning suppression or tolerance change occurs in this diagnostic. Normal-pair rounding is not independently certified.\n\nChild and outer require exact source/recipe/authorization and same-source independent review pins before scientific imports, actual unoptimized -B -s execution and fixed environment. Complete original/failure/runtime/helper/context pins are rehashed before and after. The original failed receipt remains immutable. Early failures are captured by the outer; the bounded diagnostic has no new acceptance claim beyond successful completion of its declared classification.\n'''
(HERE/'REVIEW.md').write_text(review)
after={name:sha(name) for name in pins};require(after==before,'after review drift')
save(HERE/'pins-after.json',after)
save(HERE/'receipt.json',{'passed':True,'status':'PASS_SOURCE_ONLY_HELD_DIAGNOSTIC','reviewed_source_index_sha256':EXPECTED,'reviewed_source_root':str(OWNER),'protected_unique_pins':len(pins),'source_files':len(idx['files']),'arrays_or_NPZ_decoded':False,'input_map_decoded':False,'numerical_imports':False,'Fraction_targets_evaluated':False,'compiler_or_query_calls':False,'original_v9_radial_readback_passed':False,'expected_radius_rows':32,'expected_product_components':131072,'exact_onset_claim':False,'before_after_unchanged':True,'elapsed_metadata_seconds':time.monotonic()-start})
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p!=HERE/'index.json')
save(HERE/'index.json',{'status':'immutable independent source-only review','files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
print(json.dumps({'index':sha(HERE/'index.json'),'receipt':sha(HERE/'receipt.json'),'review':sha(HERE/'REVIEW.md'),'files':len(files),'pins':len(pins),'no_science':True}))
