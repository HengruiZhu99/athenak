#!/usr/bin/env python3
"""Independent saved finite-SAT native entry readback; no new evolution."""
from pathlib import Path
import csv,json,hashlib,numpy as np
from compare_closure_field_norms import read
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/full-boundary-v4';work=root/'build-goal1-minkowski-20261018/full-boundary-v4/native-admission'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cases=['v3-control4','SAT-stationary4','SAT-pulse4','SAT-resume8'];data={c:read(work/(c+'-segment0')) for c in cases}
prior=read(root/'build-goal1-minkowski-20261018/full-canonical-v3/native-admission/full-pulse4-segment0')
assert all(np.array_equal(a[:,:,4:-4,4:-4,4:-4],b[:,:,4:-4,4:-4,4:-4]) for a,b in zip(data['v3-control4']['packs'],prior['packs']))
refs={};endpoints={};ports=[];energy=[]
for c in cases:
 folder=work/(c+'-segment0');r=data[c]['receipt'];assert r['reference_unchanged'];refs[c]={}
 for file,payload in r['reference_payloads'].items():refs[c][file]=h(folder/file);assert refs[c][file]==payload['sha256']
 rows=list(csv.DictReader((folder/'cubed_sphere_conformal.csv').open()));last=max(float(r['time']) for r in rows);endpoints[c]=[r for r in rows if float(r['time'])==last]
 if c!='v3-control4':
  p=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((folder/'cubed_sphere_boundary_sat.csv').open())];assert len(p)==12
  assert all(r['points']==384 and r['min_pivot']>0 and r['max_incoming_speed']<0 and r['min_Omega']>0 and r['port_identity_defect']<=1e-10 and r['principal_closed']<=0 for r in p)
  ports.extend([dict(case=c,**r) for r in p])
  e=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((folder/'cubed_sphere_canonical_ko_energy.csv').open())]
  assert len(e)==24 and {r['region'] for r in e}=={0.,1.}
  assert all(r['relative_identity_defect']<=1e-10 and r['direct_energy_work']<=1e-10*max(abs(r['direct_energy_work']),abs(r['negative_line_energy_work']),r['absolute_cell_work'],1e-30) for r in e)
  energy.extend([dict(case=c,**r) for r in e])
assert len({v for row in refs.values() for v in row.values()})==1
assert list(refs.values())[0]['cubed_sphere_reference_before.bin']==prior['receipt']['reference_payloads']['cubed_sphere_reference_before.bin']['sha256']
stationary=endpoints['SAT-stationary4'];drift=max(float(r['max_deviation']) for r in stationary);constraints=max(abs(float(r[k])) for r in stationary for k in ['H_l2','H_linf','M_conformal_l2','Z_conformal_l2','Theta_l2']);assert drift<=1e-9 and constraints<=1e-8
child=list(csv.DictReader((work/'SAT-resume8-segment0/cubed_sphere_conformal.csv').open()));first=min(float(r['time']) for r in child);assert endpoints['SAT-pulse4']==[r for r in child if float(r['time'])==first]
report={'native_entry_pass':True,'v3_control_active_fields_bitwise_equal':True,'all_references_before_after_match_v3':True,'reference_sha256':refs,
 'native_SAT_installed':True,
 'stationary_max_drift':drift,'stationary_max_reported_constraint':constraints,'same_method_checkpoint_continuation_exact':True,'boundary_port_rows':len(ports),'live_boundary_certificates':sum(int(r['points']) for r in ports),
 'maximum_port_identity_defect':max(r['port_identity_defect'] for r in ports),'minimum_strict_port_pivot':min(r['min_pivot'] for r in ports),'maximum_principal_closed_work':max(r['principal_closed'] for r in ports),
 'positive_signed_SAT_work_rows':sum(r['SAT_work']>0 for r in ports),'positive_affine_work_rows':sum(r['affine_work']>0 for r in ports),'effective_radius_range':[min(r['min_effective_radius'] for r in ports),max(r['max_effective_radius'] for r in ports)],
 'max_metric_derivative_trace_leakage':max(r['max_derivative_trace'] for r in ports),'dissipative_stage_energy_rows':len(energy),'max_dissipative_identity_defect':max(r['relative_identity_defect'] for r in energy),
 'boundary_ports':ports,'dissipative_energy':energy,'endpoints':endpoints,'checkpoint_payloads':{c:d['payload'] for c,d in data.items()},'binary_sha256':json.loads((out/'native-build.json').read_text())['binary_sha256'],
 'full_method_adopted':False,'exact_scri_admitted':False,'scientific_acceptance_admitted':False,'review_source_sha256':h(Path(__file__)),
 'scope':'Actual native finite-boundary residual SAT and algebraic local principal/affine port-work check, short stationary/pulse/continuation and bitwise control. Not the complete mapped discrete RHS energy, reflection/constraint absorption, resolved waveform error or exact-scri acceptance.'}
(out/'native-review.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['reference_sha256','boundary_ports','dissipative_energy','endpoints','checkpoint_payloads','scope']})
