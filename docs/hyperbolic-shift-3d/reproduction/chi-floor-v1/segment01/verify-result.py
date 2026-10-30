from pathlib import Path
import json, hashlib, struct
import numpy as np
base=Path('output/hyperbolic-core-shell/chi-floor-v1/segment01')
worker=json.loads((base/'worker-status.json').read_text())
verified=[]
for path,record in worker['cases'][0]['artifacts'].items():
 p=base/'output'/Path(path).name
 assert p.stat().st_size==record['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
 verified.append(dict(path=path,**record))
r=json.loads((base/'output/puncture.report.json').read_text())
p=base/'output/puncture.checkpoint';header=p.read_bytes()[:168]
magic,n,nf,kind,admitted,cells,surface,steps,time,dt=struct.unpack('<16s4i3Q2d',header[:72])
assert magic.rstrip(b'\0')==b'HYP3D_MPI_CF1' and (n,nf,kind,admitted)==(16,31,2,0)
assert steps==r['steps'] and time==r['t'] and dt==r['dt'] and struct.unpack('<2d',header[152:])==(1e-8,.03)
assert p.stat().st_size==168+16*(31*cells+surface)
u=np.fromfile(base/'output/puncture.failed-stage.bin',dtype='<f8',count=cells*31).reshape(cells,31)
a=np.fromfile(p,dtype='<f8',offset=168,count=cells*31).reshape(cells,31)
g=np.zeros((cells,3,3));g[:,0,0]=1+u[:,1];g[:,1,1]=1+u[:,4];g[:,2,2]=1+u[:,6];g[:,0,1]=g[:,1,0]=u[:,2];g[:,0,2]=g[:,2,0]=u[:,3];g[:,1,2]=g[:,2,1]=u[:,5]
def coords(i):
 if i>=8*n**3:return None
 b,j=divmod(int(i),n**3);k,j=divmod(j,n*n);j,ix=divmod(j,n)
 return [-.18+((b>>d)&1)*.18+(.18/n)*(v+.5) for d,v in enumerate([ix,j,k])]
def details(mask):
 ids=np.flatnonzero(mask);out=dict(count=len(ids),indices=ids.tolist(),positions=[coords(i) for i in ids])
 return out
finite=np.isfinite(g).all(axis=(1,2));eig=np.full((cells,3),np.nan);eig[finite]=np.linalg.eigvalsh(g[finite])
finite_fields=details(~np.isfinite(u).all(axis=1))
finite_fields['lapse_perturbation_values']={str(int(i)):str(u[i,18]) for i in np.flatnonzero(~np.isfinite(u[:,18]))}
finite_fields['fields']={str(int(i)):np.flatnonzero(~np.isfinite(u[i])).tolist() for i in np.flatnonzero(~np.isfinite(u).all(axis=1))}
metric=details(finite & (eig[:,0]<=0))
metric['eigenvalues']=[eig[i].tolist() for i in metric['indices']]
result=dict(t_over_M=r['t_over_M'],failure=r['error'],diagnostic_t_over_M=3*r['diagnostic_time'],checkpoint_admitted=r['checkpoint_admitted'],artifacts_verified=verified,failed_state=dict(nonpositive_chi=details(u[:,0]<=0),nonfinite_field=finite_fields,nonpositive_metric=metric),chi_floor_stage_interventions=r['chi_floor_stage_interventions'],chi_floor_acceptance_interventions=r['chi_floor_acceptance_interventions'],chi_floor_cells=r['chi_floor_cells_index_count_radius'],lapse_floor_interventions=r['lapse_floor_node_interventions'],max_null_surface=r['max_null_surface'],max_null_ghost=r['max_null_ghost'],last_rhs=r['last_rhs'])
for label,indices in [('nonfinite_field',finite_fields['indices']),('nonpositive_metric',metric['indices']),('nonpositive_chi',result['failed_state']['nonpositive_chi']['indices'])]:
 result['failed_state'][label]['last_accepted_chi']=[float(a[i,0]) for i in indices]
 result['failed_state'][label]['failed_chi']=[float(u[i,0]) if np.isfinite(u[i,0]) else str(u[i,0]) for i in indices]
(base/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts_verified','chi_floor_cells']},indent=2))
