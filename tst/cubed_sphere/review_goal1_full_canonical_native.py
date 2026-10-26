#!/usr/bin/env python3
"""Independent saved native-v3 payload/region/energy readback. No evolution."""
from pathlib import Path
import sys,json,csv,hashlib,numpy as np
from compare_closure_field_norms import read
root=Path(__file__).resolve().parents[2]
work=root/'build-goal1-minkowski-20261018/full-canonical-v3/native-admission';out=root/'docs/validation/goal1-minkowski-20261018/full-canonical-v3'
cases=['v2-control4','full-stationary4','full-pulse4','full-resume8'];data={c:read(work/(c+'-segment0')) for c in cases}
prior=read(root/'build-goal1-minkowski-20261018/canonical-ko-v2/native-admission/canonical-pulse-segment0')
equal=all(np.array_equal(a[:,:,4:-4,4:-4,4:-4],b[:,:,4:-4,4:-4,4:-4]) for a,b in zip(data['v2-control4']['packs'],prior['packs']));assert equal
refs={c:hashlib.sha256((work/(c+'-segment0/cubed_sphere_reference_before.bin')).read_bytes()).hexdigest() for c in cases}
assert len(set(refs.values()))==1 and list(refs.values())[0]==prior['receipt']['reference_payloads']['cubed_sphere_reference_before.bin']['sha256']
assert all(d['receipt']['reference_unchanged'] for d in data.values())
energy=[];endpoints={}
for c in cases:
 folder=work/(c+'-segment0');rows=list(csv.DictReader((folder/'cubed_sphere_conformal.csv').open()));last=max(float(r['time']) for r in rows);endpoints[c]=[r for r in rows if float(r['time'])==last]
 if c!='v2-control4':
  e=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((folder/'cubed_sphere_canonical_ko_energy.csv').open())]
  assert {r['region'] for r in e}=={0.,1.};energy.extend([dict(case=c,**r) for r in e])
assert len(energy)==72
assert all(r['relative_identity_defect']<=1e-10 and r['direct_energy_work']<=1e-10*max(abs(r['direct_energy_work']),abs(r['negative_line_energy_work']),r['absolute_cell_work'],1e-30) for r in energy)
stationary=endpoints['full-stationary4'];drift=max(float(r['max_deviation']) for r in stationary)
constraints=max(abs(float(r[k])) for r in stationary for k in ['H_l2','H_linf','M_conformal_l2','Z_conformal_l2','Theta_l2']);assert drift<=1e-9 and constraints<=1e-8
child=list(csv.DictReader((work/'full-resume8-segment0/cubed_sphere_conformal.csv').open()));start=min(float(r['time']) for r in child)
assert endpoints['full-pulse4']==[r for r in child if float(r['time'])==start]
r={'native_entry_pass':True,'v2_control_active_fields_bitwise_equal':equal,'all_reference_payloads_match_v2':True,'reference_sha256':refs,
 'stationary_max_drift':drift,'stationary_max_reported_constraint':constraints,'same_method_resume4_to8_endpoint_exact':True,'energy_rows':len(energy),
 'bulk_and_shell_energy_rows':{str(i):sum(r['region']==i for r in energy) for i in [0,1]},
 'max_live_all_direction_energy_identity_defect':max(r['relative_identity_defect'] for r in energy),'max_signed_energy_work':max(r['direct_energy_work'] for r in energy),
 'energy_checks':energy,'endpoints':endpoints,'checkpoint_payloads':{c:d['payload'] for c,d in data.items()},
 'binary_sha256':json.loads((out/'native-build.json').read_text())['binary_sha256'],'full_method_adopted':False,'exact_scri_admitted':False,'acceptance_matrix_admitted':False,
 'scope':'Actual coupled native all-direction full-gradient dissipative-stage identity, bitwise method control, short stationary/pulse entry and compatible continuation. No full angular-advection/source/interface/boundary/time-integrator energy, field/oracle accuracy or exact-scri acceptance.',
 'review_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'native-review.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:r[k] for k in ['native_entry_pass','energy_rows','max_live_all_direction_energy_identity_defect','stationary_max_drift','full_method_adopted']})
