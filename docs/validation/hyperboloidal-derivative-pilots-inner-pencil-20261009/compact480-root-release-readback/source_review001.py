from pathlib import Path
import ast, hashlib, json

P=Path(__file__).resolve().parent
B=P.parent/'continuum'
C=B/'native-angular-pulse-flat-derivatives-compact-root-held-20261009'
O=B/'native-angular-pulse-flat-derivatives-compact-root-launch-held-20261009'
T=B/'native-angular-pulse-flat-derivatives-timing-held-20261009'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C/'source-index.json')=='4f5a4fbbfa6c6e4af867963189f4d3a5549116e047849bf9fac8d3c96b6c9607'
assert sha(O/'source-index.json')=='0e4fcf32687ab11271b6e1a6bf90a5ba3ce9c6877b131a4cddf5564fcedafc1f'
pins={}
for directory in (C,O):
    index=json.loads((directory/'source-index.json').read_text())
    for row in index['files']:
        assert sha(row['path'])==row['sha256'],row['path']
        pins[row['path']]=row['sha256']
        if Path(row['path']).suffix=='.py': ast.parse(Path(row['path']).read_text())
recipe=json.loads((C/'comparison-recipe.json').read_text())
old=json.loads((T/'timing-recipe.json').read_text())
for key in ('a','scri_radius','geometry_r0','geometry_r1','native_amplitudes','pulse_width','precisions','height_panels','root_tolerance','maximum_root_iterations','levels','events','tolerances','required_environment','python_runtime_path','python_runtime_sha256','mpmath_python_pins'):
    assert recipe[key]==old[key],key
for name in ('derivative_core.py','analytic_jets.py','values_context.py'):
    assert (C/name).read_bytes()==(T/name).read_bytes(),name
for path,digest in {**recipe['dependency_pins'],**recipe['mpmath_python_pins'],recipe['python_runtime_path']:recipe['python_runtime_sha256']}.items():
    assert sha(path)==digest,path
    pins[path]=digest
schema=json.loads((O/'authorization-schema.json').read_text())
assert schema['expected_child_counts']=={'ray_rows':480,'group_rows':24,'checks':38344}
assert recipe['anticipated_rows']=={'timing_ray_rows':480,'group_rows':24,'checks':38344}
for path,digest in {**schema['outer_source_pins'],**schema['source_pins']}.items(): assert sha(path)==digest,path
for key in ('fresh_invocation_path','fresh_output_path'): assert not Path(schema[key]).exists(),key
report={'passed_source_metadata_and_ast_review':True,'root_full_source_and_math_review':True,'newton_monotonic_formula_reviewed':True,'outward_endpoint_signs_and_both_width_gates_reviewed':True,'original_residual_retained':True,'scientific_settings_unchanged':True,'same480_saved_row_comparison_counts_reviewed':True,'protected_dependency_pins':pins,'exact_math_files_unchanged':True,'independent_review_pending':True,'execution_admitted':False,'full_gate_held':True,'limits':'Analytic signs use fixed non-interval height quadrature; analytic branches record zero widths as formula bookkeeping. Pilot is local arithmetic consistency, not quadrature convergence, inverse map or evolution.'}
with (P/'source-review001.json').open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':True,'pins':len(pins),'receipt_sha256':sha(P/'source-review001.json')}))
