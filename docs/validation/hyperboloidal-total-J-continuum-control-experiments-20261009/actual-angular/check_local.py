"""Run staged local gates and retain every run independently."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import time

P=Path(__file__).resolve().parent
mode=sys.argv[1] if len(sys.argv)>1 else 'release'
assert mode in ('release','debug')
T=json.loads((P/'tolerances.json').read_text());S=json.loads((P/'local-supplement-tolerances.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
exe=P/('bridge-'+mode);expected=json.loads((P/('build-'+mode+'-latest.json')).read_text())
assert sha(exe)==expected['executable_sha256']
k=1
while (P/'local-attempts'/('%s-%03d'%(mode,k))).exists():k+=1
A=P/'local-attempts'/('%s-%03d'%(mode,k));A.mkdir(parents=True)
for name in ('bridge.cpp','tolerances.json','local-supplement-tolerances.json','check_local.py'):
    (A/name).write_bytes((P/name).read_bytes())
cmd=[str(exe),'--local'];start=time.monotonic();r=subprocess.run(cmd,text=True,capture_output=True)
(A/'stdout.json').write_text(r.stdout);(A/'stderr').write_text(r.stderr)
receipt={'command':cmd,'exit_code':r.returncode,'seconds':time.monotonic()-start,'executable_sha256':sha(exe),'bridge_source_sha256':sha(P/'bridge.cpp'),'tolerance_sha256':sha(P/'tolerances.json'),'supplement_tolerance_sha256':sha(P/'local-supplement-tolerances.json')}
checks={}
if r.returncode==0:
    d=json.loads(r.stdout)
    def finite(v):
        if isinstance(v,dict):return all(finite(x) for x in v.values())
        if isinstance(v,list):return all(finite(x) for x in v)
        return not isinstance(v,(int,float)) or math.isfinite(v)
    checks['all_values_finite']=finite(d)
    checks['reference_rhs']=d['reference_rhs_max']<=T['reference_raw_rhs_and_constraints_absolute_max']
    checks['reference_constraints']=d['reference_constraints_max']<=T['reference_raw_rhs_and_constraints_absolute_max']
    checks['actual_double_dual_best_amplitude']=d['double_dual_best_eps_max']<=T['double_dual_directional_fd_relative_vector_error']
    checks['actual_double_dual_all_amplitudes']=d['double_dual_all_eps_max']<=T['double_dual_directional_fd_relative_vector_error']
    checks['wrapper_negative_control']=d['double_only_wrapper_difference']>=T['double_only_wrapper_negative_control_minimum_difference']
    checks['raw22_input_normal']=d['input_normal_scaled']<=T['input_algebraic_normal_scaled_max']
    checks['raw22_output_normal']=d['output_normal_scaled']<=T['output_algebraic_normal_scaled_max']
    checks['spatial_algebraic_tangent_normals']=d['spatial_tangent_normals_scaled']<=S['full_metric_first_second_and_A_first_tangent_normals_scaled_max']
    checks['native_point_projector_identity']=d['native_point_projector_best_eps_error']<=S['native_point_projector_directional_identity_relative_max']
    checks['coordinate_zz_chart_identity']=d['coordinate_zz_chart_tangent_error']<=S['coordinate_zz_chart_tangent_identity_relative_max']
    checks['independent_solid_laplacian']=d['solid_laplacian_scaled_error']<=S['independent_solid_harmonic_laplacian_scaled_max']
    checks['core_TT_exact_oracle']=d['TT_core_max']<=T['core_TT_and_origin_oracle_absolute_max']
    checks['origin_primary_all_m']=d['origin_primary_allm_max']<=T['core_TT_and_origin_oracle_absolute_max']
    jet=max(row['errors'][-1] for row in d['actual_reference_map_jet_fd'])
    checks['actual_reference_coefficient_spatial_jets']=jet<=T['actual_reference_coefficient_jet_fd_scaled_max']
    cdot=max(max(map(abs,row['Cdot'][-1]))/max(1,row['rate_norm']) for row in d['pure_gauge'])
    checks['pure_gauge_initial_constraints']=d['initial_gauge_constraint_max']<=T['pure_gauge_initial_constraints_absolute_max']
    checks['coefficient_aware_pure_gauge_Cdot']=cdot<=T['coefficient_aware_pure_gauge_Cdot_scaled_max']
    receipt.update(checks=checks,passed_all_local_gates=all(checks.values()),reference_map_finest_scaled_error=jet,pure_gauge_finest_scaled_Cdot=cdot,full22_actual_double_dual_cases=d['fd_cases'],raw22_algebraic_lift_cases=280,reference_points=14)
else:receipt.update(passed_all_local_gates=False,checks={'process_exit_zero':False})
(A/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(P/('local-'+mode+'-latest.json')).write_text(json.dumps({'attempt':str(A),'receipt_sha256':sha(A/'receipt.json'),'passed_all_local_gates':receipt['passed_all_local_gates']},indent=2)+'\n')
print(json.dumps(receipt,indent=2))
sys.exit(0 if receipt['passed_all_local_gates'] else 1)
