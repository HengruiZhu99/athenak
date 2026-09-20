"""Exact frozen full-state exterior DtN reference, not a time-domain code BC."""
import argparse,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.model.resolve()))
import numpy as np
from scipy import linalg
from volume import Config,QP,PP,volume,schur,asdict

def exterior(lam,k,cfg):
 V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
 part=lambda v,a,b:v[np.ix_(a,b)]
 C=part(V0,QP,PP);Ci=np.linalg.inv(C);Q0=lam*np.eye(10)-part(V0,QP,QP);Q1=-part(V1,QP,QP);P0=lam*np.eye(10)-part(V0,PP,PP);P1=-part(V1,PP,PP)
 M0=P0@Ci@Q0-part(V0,PP,QP);M1=P0@Ci@Q1+P1@Ci@Q0-part(V1,PP,QP);M2=P1@Ci@Q1-part(V2,PP,QP)
 J=np.block([[np.zeros((10,10)),np.eye(10)],[-np.linalg.solve(M2,M0),-np.linalg.solve(M2,M1)]])
 T,Z,n=linalg.schur(J,output='complex',sort=lambda z:z.real< -1e-9)
 if n!=10:raise ValueError('exterior stable count '+str(n))
 W=Z[:,:10];q=W[:10];dq=W[10:];N=np.linalg.solve(q.T,dq.T).T
 return N,{'exterior_subspace_defect':float(np.linalg.norm(J@W-W@T[:10,:10])/np.linalg.norm(J@W)),'exterior_q_condition':float(np.linalg.cond(q))}

def check(z,k,cfg):
 N,meta=exterior(z,k,cfg);U,D,T,_=schur(z,k,cfg);B=D[QP]-N@U[QP];_,R=np.linalg.qr(U);Bi=B@np.linalg.inv(R);norms=np.linalg.norm(Bi,axis=1);sv=np.linalg.svd(Bi/norms[:,None],compute_uv=False)
 return dict(lambda_value=[z.real,z.imag],k=float(k),sigma_min=float(sv[-1]),**meta)
cfg=Config(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295)
old=json.load(open(a.output.parent/'screen-results.json'));points=[]
for case in old['cases']:
 if case['name']!='baseline':continue
 k=case['k']
 for r in case['roots']:points.append(('old_zero_rate_root',complex(*r['lambda_value']),k))
 z=complex(*case['physical_coupled_known_surface_mode']['lambda_value']);points.append(('old_physical_only_surface_root',z,k))
 for re in [1e-7,1e-5,.001,.01]:
  for im in [0,k/np.sqrt(2),k,2*k]:points.append(('reference_sample',re+1j*im,k))
res=[]
for tag,z,k in points:
 row=check(z,k,cfg);row['label']=tag;res.append(row)
a.output.write_text(json.dumps({'scope':'Exact Laplace-Fourier full-state exterior decaying-subspace reference at frozen coefficients, includes all original lower-order damping; no rational fit, time-domain implementation, discrete closure, or global stability proof.','config':asdict(cfg),'results':res},indent=2)+'\n')
print('points',len(res),'min_sigma',min(r['sigma_min']for r in res),'max_exterior_defect',max(r['exterior_subspace_defect']for r in res))
