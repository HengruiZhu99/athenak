"""Verify terminal evidence and inspect saved rejected tensors; no evolution."""
from pathlib import Path
import json,hashlib,struct,shutil,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
from smooth_direct_campaign import CONFIG,cells
RAW=ROOT/'output/hyperbolic-core-shell/smooth-direct-v1/aurora'
p=RAW/'campaigns/n24-smooth-direct-fresh20m-v1-segment01';out=p/'output';dst=HERE/'n24-failure'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
w=json.loads((p/'worker-status.json').read_text());r=json.loads((out/'puncture.report.json').read_text());b=json.loads((p/'binding.json').read_text());col=json.loads((p/'collected.json').read_text())
assert col['scheduler']['job_state']=='F' and w['status']=='failed' and r['status']=='failed'
assert w['source_manifest_sha256']==b['source_manifest_sha256']=='a7c265d827621328406992e1fb8f175d5a629eba6ddd618066e482c824d27b2c'
assert w['plan_sha256']==b['plan_sha256']==sha(p/'plan.json')
assert w['binaries']==b['binaries']=={'build-sycl-mpi-smooth-direct-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_smooth_direct':'1cb01219a803623d877ef9bac8a8fb3cc3619866da2ba776239ed825f7b9d942'}
case=w['cases'][0];assert not case['timed_out'] and case['storage_error'] is None
for name,a in case['artifacts'].items():
 f=RAW/name;assert f.stat().st_size==a['bytes'] and sha(f)==a['sha256'],name
cp=out/'puncture.checkpoint'
with cp.open('rb') as f:header=f.read(168)
h=struct.unpack('<16s4i3Q2d',header[:72]);assert h[:5]==(b'HYP3D_MPI_GD1'.ljust(16,b'\0'),24,31,2,0)
assert header[72:]==CONFIG and h[5]==cells(24)==r['cells'] and h[6]==6*24*24*31
assert h[9]==r['dt']==.0005*16/24 and 0<h[8]<20/3
assert b['inputs']['inputs/puncture-continuous-n24-smooth-v1.txt']=='6c265e8dc309c70a2c7017a80eddd2d76f3a185b4f1d6108cc65b8bfc85be914'
assert h[8]==r['t'] and h[7]==r['steps'] and cp.stat().st_size==168+16*(31*h[5]+h[6])
a=np.fromfile(out/'puncture.failed-stage.bin',dtype='<f8');assert len(a)==31*h[5]+h[6];a=a[:31*h[5]].reshape(-1,31)
coords=ROOT/'output/hyperbolic-core-shell/smooth-direct-v1/coordinates-n24.csv';xyz=np.genfromtxt(coords,delimiter=',',names=True);assert len(xyz)==len(a)
g=np.zeros((len(a),3,3));pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
for k,(i,j) in enumerate(pairs):g[:,i,j]=g[:,j,i]=a[:,k+1]+(i==j)
finite=np.isfinite(a).all(axis=1);minor=g[:,0,0]*g[:,1,1]-g[:,0,1]**2;det=np.linalg.det(g)
bad=np.where((~finite)|(g[:,0,0]<=0)|(minor<=0)|(det<=0))[0]
last=np.fromfile(cp,dtype='<f8',offset=168,count=31*h[5]).reshape(-1,31)
points=[]
for i in bad:
 q={k:float(xyz[k][i]) for k in ['x','y','z','r']};A=(1+q['r']**2)/2
 points.append(dict(cell=int(i),**q,finite=bool(finite[i]),determinant=float(det[i]),eigenvalues=np.linalg.eigvalsh(g[i]).tolist(),chi=float(a[i,0]),conformal_lapse=float(a[i,18]+A),previous_chi=float(last[i,0]),previous_conformal_lapse=float(last[i,18]+A)))
signs={}
for field,name in [(0,'chi'),(18,'conformal_lapse')]:
 ev=[v for v in r['candidate_sign_events'] if v[1]==field]
 signs[name]=dict(points=len(ev),negative_stage=sum(v[5] for v in ev),negative_final=sum(v[6] for v in ev),zero_stage=sum(v[7] for v in ev),zero_final=sum(v[8] for v in ev),minimum=min((v[9] for v in ev),default=None),first_negative_t_over_M=3*min(v[11] for v in ev if v[11] is not None))
summary=dict(verified_failure_evidence=True,job='9014872',error=r['error'],last_accepted_t_over_M=3*r['t'],rejected_final_candidate_t_over_M=3*(r['t']+r['dt']),last_diagnostic_t_over_M=3*r['diagnostic_time'],checkpoint_admitted=False,artifact_hashes=case['artifacts'],coordinate_sha256=sha(coords),bad_metric_cells=points,nonfinite_volume_values=int((~np.isfinite(a)).sum()),candidate_signs=signs,chi_floor_stage=r['chi_floor_stage_interventions'],chi_floor_final=r['chi_floor_acceptance_interventions'],lapse_floor_updates=r['lapse_floor_node_interventions'],scri_interpolated_negative_chi=r['scri_interpolated_chi_negative_count'],scri_interpolated_zero_chi=r['scri_interpolated_chi_zero_count'],max_null_surface=r['max_null_surface'],max_null_ghost=r['max_null_ghost'],limitation='Rejected snapshot identifies immediate invalid tensors, not the cause of their preceding growth. Counts include rejected attempts.')
(dst/'verification-and-location.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ['worker-status.json','collected.json','binding.json','plan.json']:
 shutil.copyfile(p/name,dst/name)
for name in ['puncture.report.json','puncture.history.csv','puncture.log','puncture.fastflow.horizon_summary_0.txt']:
 shutil.copyfile(out/name,dst/name)
print(json.dumps({**{k:v for k,v in summary.items() if k not in ["bad_metric_cells","artifact_hashes"]},"bad_metric_cell_count":len(points),"first_bad_metric_cells":points[:6]},indent=2))
