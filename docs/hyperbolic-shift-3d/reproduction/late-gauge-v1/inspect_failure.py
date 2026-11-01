from pathlib import Path
import hashlib,json,struct
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];raw=ROOT/'output/hyperbolic-core-shell/late-gauge-v1/aurora';folder=raw/'campaigns/n16-late-gauge-fresh10m-v1-segment01';out=HERE/'n16-failed';out.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
w=json.loads((folder/'worker-status.json').read_text());c=json.loads((folder/'collected.json').read_text());assert w['status']=='failed' and c['scheduler']['job_state']=='F'
assert digest(folder/'plan.json')==w['plan_sha256'];case=w['cases'][0]
for name,v in case['artifacts'].items():
 p=raw/name;assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256'],name
cp=folder/'output/puncture.checkpoint';h=struct.unpack('<16s4i3Q2d',cp.read_bytes()[:72]);assert h[0].rstrip(b'\0')==b'HYP3D_MPI_LG1' and h[4]==0
report=json.loads((folder/'output/puncture.report.json').read_text());assert report['status']=='failed' and not report['checkpoint_admitted'];assert h[8]==report['t'] and h[7]==report['steps']
u=np.fromfile(folder/'output/puncture.failed-stage.bin',dtype='<f8');assert len(u)==31*h[5]+h[6];volume=u[:31*h[5]].reshape(-1,31);surface=u[31*h[5]:].reshape(-1,31)
coords_path=ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv';coords=np.genfromtxt(coords_path,delimiter=',',names=True);assert len(coords)==h[5]
metrics=[]
for a in [volume,surface]:
 g=np.zeros((len(a),3,3))
 for k,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=a[:,k+1]+(i==j)
 finite=np.all(np.isfinite(g),axis=(1,2));ev=np.full((len(a),3),np.nan);ev[finite]=np.linalg.eigvalsh(g[finite]);metrics.append((g,ev,finite))
g,ev,finite=metrics[0];bad=np.flatnonzero(finite&(ev[:,0]<=0));k=32798;assert k in bad
sample=lambda j:dict(cell=int(j),coordinates=[float(coords[x][j]) for x in ['x','y','z']],radius=float(coords['r'][j]),chi=float(volume[j,0]),conformal_lapse=float(volume[j,18]+(1+coords['r'][j]**2)/2),metric=g[j].tolist(),metric_eigenvalues=ev[j].tolist(),Theta=float(volume[j,14]))
result=dict(artifact_hashes_verified=True,artifacts=case['artifacts'],checkpoint_admitted=False,last_accepted_t_over_M=report['t_over_M'],last_diagnostic_t_over_M=3*report['diagnostic_time'],failed_candidate_stage_time='not recorded; do not equate with last accepted or diagnostic time',error=report['error'],reported_cell=sample(k),bad_volume_metric_cells=len(bad),bad_volume_metric_radius_range=[float(coords['r'][bad].min()),float(coords['r'][bad].max())],nonfinite_volume_metric_cells=int(np.sum(~finite)),bad_surface_metric_cells=int(np.sum(metrics[1][1][:,0]<=0)),nonfinite_surface_metric_cells=int(np.sum(~metrics[1][2])),minimum_volume_metric_eigenvalue=float(np.nanmin(ev)),last_recorded_horizon_radius=report['horizon_radius'],max_null_surface=report['max_null_surface'],max_null_ghost=report['max_null_ghost'],floor_counts={k:report[k] for k in ['chi_floor_stage_interventions','chi_floor_acceptance_interventions','lapse_floor_node_interventions','surface_chi_floor_stage_interventions','surface_chi_floor_final_interventions','scri_interpolated_chi_negative_count','scri_interpolated_chi_zero_count']})
# The unadmitted file stores the last accepted physical state. Read it only;
# it remains categorically ineligible as a restart input.
a=np.memmap(cp,dtype='<f8',mode='r',offset=168,shape=(len(coords),31));r=coords['r'];O=(1-r*r)/2;alpha=a[:,18]+(1+r*r)/2;chi=a[:,0]
ag=np.zeros((len(a),3,3))
for f,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):ag[:,i,j]=ag[:,j,i]=a[:,1+f]+(i==j)
ae=np.linalg.eigvalsh(ag);assert np.all(ae>0)
t=np.clip((.9-r)/.5,0,1);z=np.minimum(t,1-t);s=z**5*(126+z*(-420+z*(540+z*(-315+70*z))));s=np.where(t>.5,1-s,s);F=alpha**2+4*O*O*alpha
L=chi*((1-s)*F+s*2*O*alpha)+2*s*s*np.maximum(O*O*chi,1e-5);mu=s+(1-s)*1.5*chi*F
ids=np.flatnonzero((r>.1)&(r<.95));gaps={}
for name,gap in [('transverse_shift_minus_lapse',mu-L),('longitudinal_shift_minus_lapse',4*mu/3-L),('lapse_minus_physical',L-alpha*alpha*chi)]:
 j=ids[np.argmin(gap[ids])];negative=ids[gap[ids]<0]
 gaps[name]=dict(minimum=float(gap[j]),minimum_radius=float(r[j]),negative_cells=len(negative),negative_radius_range=None if not len(negative) else [float(r[negative].min()),float(r[negative].max())])
result['last_accepted_state_audit']=dict(read_only_unadmitted_checkpoint=True,minimum_metric_eigenvalue=float(ae.min()),reported_cell_metric_eigenvalues=ae[k].tolist(),reported_cell_chi=float(chi[k]),reported_cell_conformal_lapse=float(alpha[k]),normalized_squared_speed_gaps_on_r010_095=gaps,scope='Last accepted state only. Sign changes indicate lost separation, not proof that a characteristic collision caused failure.')
result['reported_running_cfl_max']=report['max_cfl'];result['reported_last_rhs_max']=report['last_rhs']
(out/'failure-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
