#!/usr/bin/env python3
"""Review actual native true-face operator entry without claiming evolution."""
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];w=root/'build-goal1-minkowski-20261018';out=root/'docs/validation/goal1-minkowski-20261018/physical-interface-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
case=w/'physical-interface-v1/B16-A16-R24-physical';control=w/'physical-interface-v1/B8-A8-R24-control';old=w/'native-donor-operator-v1/B8-A8-R24-ungraded-attempt2'
rows=list(csv.DictReader((case/'cubed_sphere_operator.csv').open()));constant={key:max(float(x[key]) for x in rows if x['profile']=='0') for key in ['value_linf','first_linf','second_linf']};collapse=0
for profile in ['0','1','2','3']:
 for region in ['bulk','shell']:
  rs=[x for x in rows if x['profile']==profile and x['region']==region and x['subset']=='all']
  for key in ['first_l2','first_linf','second_l2','second_linf','mixed_l2','mixed_linf']:
   vals=[float(x[key]) for x in rs if not key.startswith(('second','mixed')) or x['second_tested']=='1'];collapse=max(collapse,max(vals)-min(vals))
proof=json.loads((case/'cubed_sphere_physical_interface.json').read_text());dep=json.loads((case/'cubed_sphere_physical_dependency.json').read_text());control_equal=(old/'cubed_sphere_operator.csv').read_bytes()==(control/'cubed_sphere_operator.csv').read_bytes();refs={};terminal=True
for name in [case,control]:
 receipt=json.loads((name/'receipt.json').read_text());terminal&=receipt['exit_code']==0;refs[name.name]={p:sha(name/p) for p in ['cubed_sphere_reference_before.bin','cubed_sphere_reference_after.bin']}
reference_equal=all(len(set(x.values()))==1 for x in refs.values());protocol=json.loads((out/'native-protocol-v1.json').read_text());budgets=protocol['budgets']
passed=bool(control_equal and reference_equal and terminal and proof['native_operator_enabled'] and proof['energy_identity_scaled']<=budgets['physical_mass_SBP_identity_scaled'] and constant['value_linf']<=budgets['value_absolute'] and max(constant['first_linf'],constant['second_linf'])<=budgets['constant_first_second_absolute'] and collapse<=budgets['field_slot_coefficient_collapse'] and dep['actual_shell_first_response_to_bulk']>budgets['bulk_to_shell_first_response_min'])
report={'scope':__doc__,'native_operator_entry_pass':passed,'control_operator_CSV_bitwise_identical':control_equal,'constant_maxima':constant,'field_slot_normalized_norm_max_range':collapse,'actual_native_physical_operator':proof,'actual_native_dependency':dep,'reference_before_after_sha256':refs,'all_reference_payloads_unchanged':reference_equal,'rows':len(rows),'first_derivative_slots':22,'second_derivative_slots':16,'all_handles_terminal':terminal,'coupled_evolution_trajectories':0,'full_coupled_stability_admitted':False,'outer_true_face_SAT_implemented':False,'exact_scri_implemented':False,'large_nonconstant_derivative_errors_retained':True,'source_sha256':sha(Path(__file__)),'native_receipt_sha256':sha(case/'receipt.json')};(out/'native-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='reference_before_after_sha256'},indent=2));assert passed
