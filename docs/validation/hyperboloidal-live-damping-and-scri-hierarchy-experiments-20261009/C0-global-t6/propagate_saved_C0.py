"""Fresh exploratory t6 action of pinned C0 matrices; no native/canonical rerun."""
from pathlib import Path
import argparse,hashlib,json,time,traceback
import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import expm
w=Path(__file__).resolve().parent;old=w.parent/'full-tensor-propagator/full22-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);a=p.parse_args();g=a.gauge
matrix=old/f'{g}-projected-J20.npz';seed=old.parent/f'{g}-validation-vectors.npz'
expected={'production':'bfb62f29aaa426f173aa75ddcfa0c86d9784c0542ea4a2a6b69debb5a85eb22a','spatialnorm':'767b7c998e27db4d598e260f80e2181afe31d427c3f58c9a7bb60e30c35dede6'}
assert sha(matrix)==expected[g]
assert sha(seed)==sha(old.parent/'projected-v1'/seed.name)
A=load_npz(matrix);vs=dict(np.load(seed));names=['gauge_pulse','shell_random']
times=np.linspace(0,6,241);result=np.full((len(times),A.shape[0],2),np.nan);completed=[1,1]
for col,name in enumerate(names):result[0,:,col]=vs[name]/np.linalg.norm(vs[name])
receipts=[];t0=time.monotonic();matvecs=0;residual_matvecs=0;trial=np.zeros(A.shape[0]);guard=1e12
pins={'matrix_path':str(matrix),'matrix_sha256':sha(matrix),'seed_path':str(seed),'seed_sha256':sha(seed),'driver_sha256':sha(Path(__file__)),'original_t2_canonical_agreement_sha256':sha(old/'canonical-vs-krylov-all-states.json')}
def basis(v):
 global matvecs
 beta=np.linalg.norm(v);V=np.zeros((len(v),81),order='F');H=np.zeros((81,80));V[:,0]=v/beta;k=80
 for j in range(80):
  z=A@V[:,j];matvecs+=1;initial=np.linalg.norm(z)
  for _ in range(2):
   q=V[:,:j+1].T@z;H[:j+1,j]+=q;z-=V[:,:j+1]@q
  H[j+1,j]=np.linalg.norm(z)
  if H[j+1,j]<1e-13*max(1.,initial):k=j+1;break
  V[:,j+1]=z/H[j+1,j]
 return beta,V[:,:k],H[:k,:k],k
def action(beta,V,H,k,dt):
 c=min(50,k);coef=expm(dt*H)[:,0];f=beta*(V@coef);q=beta*(V[:,:c]@expm(dt*H[:c,:c])[:,0])
 return f,float(np.linalg.norm(f-q)/max(np.linalg.norm(f),1e-300)),coef
def check_guard(q):
 if not np.isfinite(q).all():raise RuntimeError('nonfinite attempted vector; retained in guard NPZ')
 norm=float(np.linalg.norm(q))
 if not np.isfinite(norm) or norm>guard:raise RuntimeError('Euclidean amplification guard exceeded1e12; attempted vector retained')
def defect(beta,V,H,coef,q):
 global residual_matvecs
 rhs=A@q;residual_matvecs+=1;r=beta*(V@(H@coef))-rhs
 return {'absolute_l2':float(np.linalg.norm(r)),'relative_state_l2_per_time':float(np.linalg.norm(r)/max(np.linalg.norm(q),1e-300)),'relative_rhs_l2':float(np.linalg.norm(r)/max(np.linalg.norm(rhs),1e-300))}
try:
 for col,name in enumerate(names):
  v=result[0,:,col].copy();t=0.;nextout=1;steps=[];begin=time.monotonic()
  while t<6-1e-13:
   beta,V,H,k=basis(v);dt=min(.1,6-t);rejections=0
   while True:
    trial,err,coef=action(beta,V,H,k,dt);check_guard(trial);inside=[];maxerr=err
    for index in range(nextout,len(times)):
     if times[index]>t+dt+1e-12:break
     if times[index]<t-1e-12:raise RuntimeError('missing Krylov output')
     q,e,c=action(beta,V,H,k,times[index]-t);check_guard(q);inside.append((index,q,c));maxerr=max(maxerr,e)
    if maxerr<=1e-10:break
    dt/=2;rejections+=1
    if dt<1e-8:raise RuntimeError('truncation-pair tolerance step underflow')
   d=defect(beta,V,H,coef,trial);outputs=[]
   for index,q,c in inside:
    result[index,:,col]=q;nextout=index+1;completed[col]=nextout
    outputs.append({'time':float(times[index]),**defect(beta,V,H,c,q)})
   t+=dt;v=trial;steps.append({'time':t,'dt':dt,'coarse_fine_max_relative_difference':maxerr,'basis_size':k,'rejections':rejections,'accepted_curve_defect':d,'output_curve_defects':outputs})
   if len(steps)%10==0 or t>=6-1e-13:print(g,name,'t',t,'steps',len(steps),'matvecs',matvecs,'norm',np.linalg.norm(v),'seconds',time.monotonic()-begin,flush=True)
  assert nextout==len(times)
  receipts.append({'name':name,'seconds':time.monotonic()-begin,'steps':steps,'final_euclidean_amplification':float(np.linalg.norm(v))})
except Exception as e:
 np.savez_compressed(w/f'{g}-stopped-guard-or-error.npz',times=times,values=result,names=np.asarray(names),completed_output_count=np.array(completed),attempted_trial=trial)
 failure={'status':'stopped_guard_or_error','error':str(e),'traceback':traceback.format_exc(),'current_seed':name,'current_time':t,'completed_output_counts':completed,'matvecs':matvecs,'residual_matvecs':residual_matvecs,'seconds':time.monotonic()-t0,'pins':pins,'partial_previous_columns':receipts,'current_steps':steps,'uncomputed_values_in_npz':'explicit NaN placeholders; attempted trial retained without repair'}
 (w/f'{g}-stopped-guard-or-error.json').write_text(json.dumps(failure,indent=2,allow_nan=False)+'\n');raise
label=f'{g}-projected-krylov-m50-80-h0.1-t6.0'
np.savez_compressed(w/(label+'.npz'),times=times,values=result,names=np.asarray(names))
r={'status':'completed','gauge':g,'semantics':'exploratory exp(t PJ22Lift), no independent t6 canonical action; empirical truncation pair and actual Krylov curve residuals are not nonnormal forward-error bounds','pins':pins,'coarse':50,'fine':80,'rtol':1e-10,'max_step':.1,'stop':6,'guard_euclidean_amplification':guard,'matvecs':matvecs,'residual_matvecs':residual_matvecs,'seconds':time.monotonic()-t0,'columns':receipts}
(w/(label+'.json')).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('done',g,label,'seconds',r['seconds'],flush=True)
