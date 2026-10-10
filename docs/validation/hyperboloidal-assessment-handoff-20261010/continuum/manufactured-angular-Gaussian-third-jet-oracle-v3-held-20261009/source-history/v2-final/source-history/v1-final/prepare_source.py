"""Metadata/AST preparation only. Never imports the held numerical modules."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-native-inverse-held-20261009'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    if (HERE/'source-index.json').exists():raise RuntimeError('one-shot source freeze')
    old=load(OLD/'recipe.json')
    protected=dict(old['pins'])
    context=[OLD/'source-index.json',OLD/'recipe.json',OLD/'attempt001/receipt.json',OLD/'attempt001/result.json',
       OLD/'outer-invocation001/receipt.json',
       ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-native-inverse-independent-review-v2-20261009/index.json',
       ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-native-inverse-independent-review-v2-20261009/receipt.json',
       ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-jets-RHS-plan-held-20261009/index.json',
       ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-jets-RHS-plan-held-20261009/DERIVATION.md',
       ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-jets-RHS-plan-held-20261009/PLAN.md',
       ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-jets-RHS-independent-definition-review-20261009/index.json',
       ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-jets-RHS-independent-definition-review-20261009/GATE-DEFINITIONS.md',
       ROOT/'build-layer-research/continuum/manufactured-angular-time-wave-compact-ADM-pencil-20261009/index.json',
       ROOT/'build-layer-research/continuum/manufactured-angular-time-wave-compact-ADM-pencil-20261009/DERIVATION.md',
       ROOT/'build-layer-research/continuum/manufactured-angular-time-wave-compact-ADM-pencil-20261009/LAYER-TIMELIKE-REDUCTION.md',
       ROOT/'build-layer-research/continuum/reference-wave-map-gauge-20261009/immutable-local-reference-wave-map-20261009/reference_wave_map.hpp',
       ROOT/'build-layer-research/continuum/reference-wave-map-26-context-independent-saved-review-20261009/index.json',
       ROOT/'build-layer-research/continuum/reference-wave-map-26-context-independent-saved-review-20261009/receipt.json']
    for p in context:protected[str(p)]=sha(p)
    for p,h in protected.items():
        if sha(p)!=h:raise RuntimeError('external pin drift '+p)
    if sha(HERE/'values_context.py')!='89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7':
        raise RuntimeError('copied scalar-height context changed')
    original=[x['value'] for x in old['native_radii']]
    radii=original[:]
    for value in ('49999/1000000','50001/1000000','949999/1000000','950001/1000000'):
        if value in radii:raise RuntimeError('unexpected duplicate radius')
        radii.append(value)
    # Ordering preserves all old radial cases, then the four endpoint neighbors.
    negative='negative/sigma1/2/eps3/4/r3/4/t1/10/p1/2'
    timing=['eps0/r0/t0/origin','eps3/4/r0/t1/10/origin','eps3/4/r1/40/t1/10/111',
       'eps3/4/r1/5/t0/p=-1/2','eps3/4/r1/2/t1/10/2,-3,6',
       'eps3/4/r949999/1000000/t2/p=1/2','eps3/4/r950001/1000000/t2/111',
       'eps3/4/r49/50/t6/p=1/8','eps3/4/r999999999999999999/1000000000000000000/t1/10/2,-3,6',negative]
    roots=lambda absolute,width:{'absolute':absolute,'width':width,'newton':16,'bisection':512}
    recipe={'status':'HELD source-only analytic Gaussian third-jet oracle; root stage release required',
       'execution_admitted':False,'layer':old['layer'],'radii':radii,'times':['0','1/10','2','6'],
       'angular_p':old['p_values'],'epsilon':['0','3/4'],'sigma':'7/20',
       'levels':[{'digits':110,'roots':roots('1e-85','1e-90')},{'digits':150,'roots':roots('1e-125','1e-130')}],
       'unit_digits':80,'unit_height_order':256,'unit_roots':roots('1e-60','1e-65'),
       'height_order':256,'height_comparison_order':128,'height_tolerance':'1e-30',
       'identity_tolerance':'1e-55','component_tolerance':'1e-55','graph_radius':'0.98',
       'records_per_level':2505,'timing_keys':timing,'stage_seconds':{'units':60,'timing':600,'full':14400},
       'outer_seconds':{'units':120,'timing':660,'full':14460},'payload_byte_cap':17179869184,
       'failure_summary_cap':1000,'expected_counts':{'units':318,'full_records':5010,'full_identity_checks':3330320,
       'full_component_checks':470752,'height_context_checks':50,'timing_records':20,'nominal_native_eligible_valid_cases':1880},
       'python_runtime_path':old['python_runtime_path'],'mpmath_parent':old['mpmath_parent'],
       'mpmath_init':old['mpmath_init'],'environment':old['environment'],'protected_inputs':protected,
       'numeric_modules_imported_in_preparation':False,'native_queries_admitted':False,
       'native_binding':'future unchanged physical-reference RWM only; no compound inner BM',
       'native_coefficient_error_protocol_admitted':False,'full_release_requires_measured_timing_review':True,
       'all_jsonl_npz_npy_large_payload_regardless_size':True}
    save(HERE/'recipe.json',recipe)
    modules=[]
    for p in sorted(HERE.glob('*.py')):
        tree=ast.parse(p.read_text(),filename=str(p))
        modules.append({'path':str(p),'sha256':sha(p),'AST_parse':True,
            'imports':[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]})
    count={'source_only':True,'candidate_imports':False,'numeric_calls':False,'compiler_calls':False,
       'registry_counts_by_integer_counting':{'radii':25,'points_per_epsilon':1252,'valid_per_level':2504,'negative_per_level':1,
        'levels':2,'total_records':5010,'identity_checks':3330320,'component_checks':470752,'units':318,'timing_records':20},
       'protected_external_inputs':len(protected),'modules':modules,'historical_WIP_executed':False,
       'identity_count_change_from_unfrozen_WIP':'ten explicit physical Ricci rows per valid record; full5010 registry unchanged',
       'reference_independence':'embedding connection versus independent radial ADM metric',
       'graph_independence':'graph-chart implicit3 composed back through full Q(x) third jets'}
    save(HERE/'source-preparation.json',count)
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='source-index.json')
    source_map={str(p):sha(p) for p in files}
    save(HERE/'source-index.json',{'status':'immutable SOURCE-ONLY held Gaussian analytic oracle candidate',
         'source_only':True,'execution_admitted':False,'file_count':len(files),'files':source_map})
    for p,h in source_map.items():
        if sha(p)!=h:raise RuntimeError('final source drift')
    for p,h in protected.items():
        if sha(p)!=h:raise RuntimeError('final external drift')
    print(json.dumps({'source_index_sha256':sha(HERE/'source-index.json'),'recipe_sha256':sha(HERE/'recipe.json'),
       'driver_sha256':sha(HERE/'run_oracle.py'),'outer_sha256':sha(HERE/'outer_once.py'),
       'files':len(files),'external_pins':len(protected),'candidate_imports':False},sort_keys=True))

if __name__=='__main__':main()
