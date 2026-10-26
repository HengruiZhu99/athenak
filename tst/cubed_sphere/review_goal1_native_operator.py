#!/usr/bin/env python3
"""Readback of actual donor/operator pilot and mandatory transmission gate."""
from pathlib import Path
import json,csv,hashlib
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/native-donor-operator-v1';case=root/'build-goal1-minkowski-20261018/native-donor-operator-v1/B8-A8-R24-ungraded-attempt2'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=json.loads((case/'receipt.json').read_text());assert receipt['exit_code']==0 and receipt['reference_unchanged']
for file,payload in receipt['payloads'].items():assert h(case/file)==payload['sha256']
rows=list(csv.DictReader((case/'cubed_sphere_operator.csv').open()));assert len(rows)==880
value=max(float(r['value_linf']) for r in rows);constant=max(float(r[k]) for r in rows if r['profile']=='0' for k in ['first_linf','second_linf','mixed_linf']);negative=max(float(r['omitted_Hessian_linf']) for r in rows)
dep=json.loads((case/'cubed_sphere_operator_dependency.json').read_text());coupling=dep['bulk_sourced_shell_rows_active_radial']>0 and max(dep[k] for k in ['shell_value_response_to_bulk_only','shell_first_response_to_bulk_only','shell_second_response_to_bulk_only'])>0
selected=[{**{k:r[k] for k in ['profile','region','subset','component','second_tested','cells']},**{k:float(r[k]) for k in ['volume','value_l2','first_l2','first_linf','second_l2','second_linf','mixed_l2','mixed_linf']}} for r in rows if r['profile']=='3' and r['component']=='0']
report={'scope':'Actual native field-slot/Cartesian-jet/donor functional pilot and structural interface dependency; no physical or nonlinear evolution.',
 'functional_controls_pass':value<=1e-12 and constant<=1e-10 and negative>=.001,'max_active_value_error':value,'max_constant_derivative_error':constant,'omitted_Hessian_negative_maximum':negative,
 'rows':len(rows),'first_derivative_field_slots':22,'second_derivative_field_slots':16,'nonpolynomial_component0_regions':selected,'dependency':dep,
 'bulk_to_shell_transmission_gate_pass':coupling,'higher_grid_extension_launched':False,'stop_reason':'Zero bulk-sourced donor targets in active radial range and exactly zero shell jet response; repair incoming Cartesian-shell coupling before refinement.',
 'native_evolution_trajectories':0,'reference_unchanged':True,'original_setup_failure_retained':True,'receipt_sha256':h(case/'receipt.json'),'review_source_sha256':h(Path(__file__))}
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['nonpolynomial_component0_regions']})
