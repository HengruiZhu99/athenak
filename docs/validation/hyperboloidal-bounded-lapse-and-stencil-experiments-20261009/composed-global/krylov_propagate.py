"""Independent adaptive Arnoldi action; truncation-pair checks, no ARPACK."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import load_npz
from scipy.linalg import expm
w=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);p.add_argument('--stop',type=float,default=2.);p.add_argument('--coarse',type=int,default=50);p.add_argument('--fine',type=int,default=80);p.add_argument('--rtol',type=float,default=1e-10);p.add_argument('--max-step',type=float,default=.1);a=p.parse_args();g=a.gauge;A=load_npz(w/f'{g}-projected-J20.npz');vs=dict(np.load(w/f'{g}-validation-vectors.npz'));names=['gauge_pulse','shell_random'];times=np.linspace(0,a.stop,int(round(a.stop/.025))+1);result=np.empty((len(times),A.shape[0],2));receipts=[];t0=time.monotonic();matvecs=0
# Two-pass modified Gram-Schmidt using matrix-vector operations.
def basis(v):
 global matvecs
 beta=np.linalg.norm(v);V=np.zeros((len(v),a.fine+1),order='F');H=np.zeros((a.fine+1,a.fine));V[:,0]=v/beta;k=a.fine
 for j in range(a.fine):
  z=A@V[:,j];matvecs+=1;initial=np.linalg.norm(z)
  for _ in range(2):
   q=V[:,:j+1].T@z;H[:j+1,j]+=q;z-=V[:,:j+1]@q
  H[j+1,j]=np.linalg.norm(z)
  if H[j+1,j]<1e-13*max(1.,initial):k=j+1;break
  V[:,j+1]=z/H[j+1,j]
 return beta,V[:,:k],H[:k,:k],k

def action(beta,V,H,k,dt):
 c=min(a.coarse,k);f=beta*(V@expm(dt*H)[:,0]);q=beta*(V[:,:c]@expm(dt*H[:c,:c])[:,0]);return f,float(np.linalg.norm(f-q)/max(np.linalg.norm(f),1e-300))
for col,name in enumerate(names):
 v=vs[name].copy();v/=np.linalg.norm(v);result[0,:,col]=v;t=0.;nextout=1;steps=[];begin=time.monotonic()
 while t<a.stop-1e-13:
  beta,V,H,k=basis(v);dt=min(a.max_step,a.stop-t);rejections=0
  while True:
   trial,err=action(beta,V,H,k,dt);inside=[];maxerr=err
   for index in range(nextout,len(times)):
    if times[index]>t+dt+1e-12:break
    if times[index]<t-1e-12:raise RuntimeError('missing Krylov output')
    q,e=action(beta,V,H,k,times[index]-t);inside.append((index,q));maxerr=max(maxerr,e)
   if maxerr<=a.rtol:break
   dt/=2;rejections+=1
   if dt<1e-8:raise RuntimeError('Krylov action failed truncation-pair tolerance')
  for index,q in inside:result[index,:,col]=q;nextout=index+1
  t+=dt;v=trial;steps.append({'time':t,'dt':dt,'coarse_fine_max_relative_difference':maxerr,'basis_size':k,'rejections':rejections})
  if len(steps)%5==0 or t>=a.stop-1e-13:print(g,name,'t',t,'steps',len(steps),'mv',matvecs,'norm',np.linalg.norm(v),'seconds',time.monotonic()-begin,flush=True)
 assert nextout==len(times)
 receipts.append({'name':name,'seconds':time.monotonic()-begin,'steps':steps,'final_euclidean_amplification':float(np.linalg.norm(v))})
label=f'{g}-projected-krylov-m{a.coarse}-{a.fine}-h{a.max_step}-t{a.stop}';np.savez_compressed(w/(label+'.npz'),times=times,values=result,names=np.asarray(names));(w/(label+'.json')).write_text(json.dumps({'gauge':g,'semantics':'exp(t PJ22Lift) via independent adaptive Arnoldi; coarse/fine truncation difference is empirical error control, not a rigorous nonnormal forward-error bound','coarse':a.coarse,'fine':a.fine,'rtol':a.rtol,'max_step':a.max_step,'stop':a.stop,'matvecs':matvecs,'seconds':time.monotonic()-t0,'columns':receipts},indent=2)+'\n');print('done',g,label,'seconds',time.monotonic()-t0,flush=True)
