#!/usr/bin/env python3
"""One-shot standard-library source/recipe pin preparation; no science."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
PLAN=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-plan-held-20261009'
PRINCIPAL=ROOT/'build-layer-research/continuum/inner-joint-principal-gate-v3-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/inner-joint-nonlinear-plan-independent-review-20261009'
RUNTIME=ROOT/'build-layer-research/continuum/einstein-reference-jet-oracle-20261009/runtime-and-execution.json'

def pin(path):
    path=Path(path).resolve();data=path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
def read(path):return json.loads(Path(path).read_text())
def write_new(name,data):
    path=HERE/name
    if path.exists():raise RuntimeError('Refuse overwrite '+str(path))
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')

def main():
    if (HERE/'source-index.json').exists():raise RuntimeError('Source already indexed')
    old=read(PLAN/'source-index.json');pins={}
    def add(row):
        actual=pin(row['path'])
        if actual['sha256']!=row['sha256']:raise RuntimeError('Input drift '+row['path'])
        if 'bytes'in row and actual['bytes']!=row['bytes']:raise RuntimeError('Size drift')
        pins[actual['path']]=actual
    for row in read(PLAN/'input-pins.json')+old['files']+[pin(PLAN/'source-index.json')]:add(row)
    for name in ['PLAN.md','ALGEBRA.md','ORACLES.md','cases.json','recipe.json','source-index.json','source-preparation.json','input-pins.json']:
        if pin(HERE/'held-plan'/name)['sha256']!=pin(PLAN/name)['sha256']:
            raise RuntimeError('Held plan copy drift')
    review=read(REVIEW/'index.json')
    if pin(REVIEW/'index.json')['sha256']!='5fdd5f3b46a4d068a96716505eada7b3aedaebfe3ba7561b9a029192d18a63df':
        raise RuntimeError('Review index mismatch')
    add(pin(REVIEW/'index.json'))
    for row in review['files']:
        path=Path(row['path'])
        if not path.is_absolute():path=REVIEW/path
        add({'path':str(path),'sha256':row['sha256']})
    review_receipt=read(REVIEW/'receipt.json')
    if not (review_receipt['status'].startswith('PASS') and
            review_receipt['inputs_unchanged'] and not review_receipt['corrections']):
        raise RuntimeError('Plan review failed')
    runtime=read(RUNTIME);add(pin(RUNTIME))
    python=runtime['python_executable'];add({'path':python,'sha256':runtime['python_executable_sha256']})
    for row in runtime['mpmath_python_sources']:add(row)
    mpmath_init=next(row['path'] for row in runtime['mpmath_python_sources'] if row['path'].endswith('/mpmath/__init__.py'))
    stdlib=Path(python).parent.parent/'lib/python3.9'
    if not stdlib.is_dir():raise RuntimeError('Exact Python stdlib path missing')
    for path in stdlib.rglob('*.py'):
        if 'site-packages' not in path.parts and '__pycache__' not in path.parts:add(pin(path))
    for path in (stdlib/'lib-dynload').glob('*.so'):add(pin(path))
    compiler_recipe=read(PRINCIPAL/'recipe.json')
    add(compiler_recipe['compiler'])
    for name in ['reference_wave_map.hpp','test_support.hpp','wide_arithmetic.hpp','dual_helpers.hpp','nonlinear_values.hpp']:
        source=ROOT/'build-layer-research/continuum/reference-wave-map-gauge-20261009/immutable-local-reference-wave-map-20261009'/name
        if pin(HERE/'inputs'/name)['sha256']!=pin(source)['sha256']:raise RuntimeError('Frozen source copy drift '+name)
    guard_inputs=sorted(pins.values(),key=lambda row:row['path'])
    write_new('input-pins.json',guard_inputs)
    modes=['reference','sources','coefficients','core-witnesses','principal','duals','invalid','nonrepresentable']
    flags={}
    for build in ['release','debug']:
        flags[build]=['-I'+str(HERE/'inputs'),'-I'+str(HERE)]+compiler_recipe[build+'_flags']
    recipe={
      'source_only':True,'execution_admitted':False,
      'scope':'finite-Omega local inner nonlinear helper only',
      'repository':str(ROOT),'source_index':str(HERE/'source-index.json'),
      'input_pins':str(HERE/'input-pins.json'),
      'held_plan_index':str(PLAN/'source-index.json'),
      'held_plan_sha256':pin(PLAN/'source-index.json')['sha256'],
      'principal_receipt':str(PRINCIPAL/'attempts/gate001/receipt.json'),
      'plan_independent_review_index':pin(REVIEW/'index.json'),
      'compiler':compiler_recipe['compiler']['path'],'python':python,
      'mpmath_parent':str(Path(mpmath_init).parent.parent),
      'mpmath_init':str(Path(mpmath_init).resolve()),
      'compile_flags':flags,'attempt_names':{'release':'Release001','debug':'Debug001'},
      'modes':modes,'expected_record_counts':{
         'reference':336,'source':4704,'coefficient':1452,'coefficient-dual':160,
         'core-witness':144,'principal':6384,'dual':2520,
         'invalid-coefficient':16,'invalid-source':6,'nonrepresentable':18},
      'high_contrast_families':['collapsed','small-alpha-large-chi','large-alpha-small-chi','chi-gradient-contrast'],
      'gauge_raw22_indices':[0,4,5,6],
      'geometry_raw22_indices':[1,2,3]+list(range(7,22)),
      'raw22_order':['alpha','chi','P','Theta','beta_x','beta_y','beta_z','gxx','gxy','gxz','gyy','gyz','gzz','Axx','Axy','Axz','Ayy','Ayz','Azz','Lambda_x','Lambda_y','Lambda_z'],
      'live_kappa1':'10/alpha','reference_kappa1':'10','kappa2':'0',
      'no_upwind_KO_ghost_RK_or_global_stencil_in_this_point_gate':True,
      'thresholds':read(PLAN/'recipe.json')['thresholds'],
      'core_large_A_oracle':'independently simplified G0*Lambda; no240/280-digit large cancellation claim',
      'precision':{'ordinary':[80,110],'main_high_contrast':[240,280],'core_closed':[110]},
      'generic_scalar_contract':'double and registered field dual D at fixed external reference position; nonzero p.radius tangent explicitly rejected',
      'environment':{'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','OMP_NUM_THREADS':'1'},
      'python_flags':['-I','-B'],'require_unoptimized_python':True,
      'no_production_native_BH_operator_spectrum_or_evolution':True,
      'outer_invocation_capture_required':'parent must retain command/env/stdout/stderr/returncode even for pre-attempt guards',
    }
    write_new('recipe.json',recipe)
    write_new('authorization-schema.json',{'local_nonlinear_helper_execution_admitted':False,'source_index_sha256':'EXACT_RELEASE_REQUIRED','recipe_sha256':pin(HERE/'recipe.json')['sha256'],'allowed_builds':['release','debug'],'scope':'local helper/source/dual fixed query suite only; no native/global'})
    write_new('source-preparation.json',{
      'source_only':True,'execution_admitted':False,
      'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'protected_inputs':len(guard_inputs),'protected_inputs_unchanged':True,
      'held_plan_copies_byte_identical':True,'frozen_RWM_copies_byte_identical':True,
      'independent_plan_review_passed':True,
      'no_scientific_import_syntax_compile_query_array_load_execution':True,
      'runtime_scope':'exact interpreter binary, all actual Python3.9 stdlib .py except site-packages/__pycache__, lib-dynload .so, all pinned mpmath .py, existing compiler/SDK/production dependency pins',
      'implementation_status':'uncompiled and unexecuted source proposal only',
    })
    # A mechanical before/after readback, not a scientific source gate.
    for row in guard_inputs:add(row)
    files=[pin(path) for path in sorted(HERE.rglob('*')) if path.is_file()]
    write_new('source-index.json',{'source_only':True,'execution_admitted':False,'scope':'unexecuted private inner nonlinear helper/probe/oracle/runner source proposal001','files':files,'file_count':len(files),'protected_inputs':len(guard_inputs),'inputs_unchanged':True,'held_plan_sha256':pin(PLAN/'source-index.json')['sha256']})
    print(json.dumps({'source_index':pin(HERE/'source-index.json'),'files':len(files),'protected_inputs':len(guard_inputs),'execution_admitted':False},sort_keys=True))

if __name__=='__main__':main()
