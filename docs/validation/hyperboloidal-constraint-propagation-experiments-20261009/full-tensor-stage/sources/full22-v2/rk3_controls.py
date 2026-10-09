"""Short-time exact native final-only linearized RK3 vs projected RK3."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import load_npz
w=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);p.add_argument('--stop',type=float,default=.01);a=p.parse_args();g=a.gauge;prefix=w/f'{g}-cache0.0001';meta=json.loads(Path(str(prefix)+'-metadata.json').read_text());N=meta['points'];A=load_npz(str(prefix)+'-J22.npz');J=load_npz(w/f'{g}-projected-J20.npz');L=np.fromfile(str(prefix)+'-lift.bin',dtype='<f8').reshape(N,22,20);P=np.fromfile(str(prefix)+'-restrict.bin',dtype='<f8').reshape(N,20,22)
def lift(v):return np.einsum('pij,pjk->pik',L,v.reshape(N,20,-1)).reshape(N*22,-1)
def project(v):return np.einsum('pij,pjk->pik',P,v.reshape(N,22,-1)).reshape(N*20,-1)
def step(v,dt,full):
 x=lift(v) if full else v;op=A if full else J;k1=op@x;k2=op@k1;k3=op@k2;y=x+dt*k1+dt*dt*k2/2+dt**3*k3/6;return project(y) if full else y
vs=dict(np.load(w.parent/f'{g}-validation-vectors.npz'));names=['gauge_pulse','shell_random'];B=np.column_stack([vs[n] for n in names]);B/=np.linalg.norm(B,axis=0)[None,:];steps=int(np.ceil(a.stop/(.03*meta['min_omega'])));dt=a.stop/steps;res={'gauge':g,'stop':a.stop,'base_steps':steps,'base_dt':dt,'nominal_pole_dt':.03*meta['min_omega'],'names':names,'runs':[]};saved={'initial':B};t0=time.monotonic()
for label,full,n in [('native_final_only',True,steps),('native_final_only_half',True,2*steps),('projected_every_RHS',False,steps)]:
 v=B.copy();t=time.monotonic()
 for k in range(n):v=step(v,a.stop/n,full)
 saved[label]=v;row={'label':label,'steps':n,'dt':a.stop/n,'seconds':time.monotonic()-t,'euclidean_amplification':np.linalg.norm(v,axis=0).tolist()};res['runs'].append(row);print(g,row,flush=True)
res['relative_dt_vs_half']=np.linalg.norm(saved['native_final_only']-saved['native_final_only_half'],axis=0).tolist();res['relative_native_vs_projected']=np.linalg.norm(saved['native_final_only']-saved['projected_every_RHS'],axis=0).tolist();res['total_seconds']=time.monotonic()-t0;np.savez_compressed(w/f'{g}-rk3-t{a.stop}.npz',**saved);(w/f'{g}-rk3-t{a.stop}.json').write_text(json.dumps(res,indent=2)+'\n');print('done',g,res,flush=True)
