"""Frozen full-state DtN stable-pole prototype; no production boundary."""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy import linalg,optimize
MODEL=Path(__file__).parent/'model'
sys.path.insert(0,str(MODEL))
from volume import Config,QP,PP,volume,schur,asdict

def exterior(lam,k,cfg):
 V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
 part=lambda v,a,b:v[np.ix_(a,b)]
 C=part(V0,QP,PP);Ci=np.linalg.inv(C);Q0=lam*np.eye(10)-part(V0,QP,QP);Q1=-part(V1,QP,QP);P0=lam*np.eye(10)-part(V0,PP,PP);P1=-part(V1,PP,PP)
 M0=P0@Ci@Q0-part(V0,PP,QP);M1=P0@Ci@Q1+P1@Ci@Q0-part(V1,PP,QP);M2=P1@Ci@Q1-part(V2,PP,QP)
 J=np.block([[np.zeros((10,10)),np.eye(10)],[-np.linalg.solve(M2,M0),-np.linalg.solve(M2,M1)]])
 T,Z,n=linalg.schur(J,output='complex',sort=lambda z:z.real< -1e-11)
 if n!=10:raise ValueError('exterior stable count '+str(n))
 W=Z[:,:10];q=W[:10];dq=W[10:];N=np.linalg.solve(q.T,dq.T).T
 return N

class Fit:
 def __init__(self,k,cfg,realcount=32,complexcount=32):
  self.k=k;self.cfg=cfg
  # Asymptotic leading term; Richardson removes O(lambda^-1) intercept error.
  f1=exterior(1e3,k,cfg);f2=exterior(2e3,k,cfg);f4=exterior(4e3,k,cfg)
  self.A=(f4-2*f2+f1)/1e3
  self.B=2*(2*f2-f4)-(2*f1-f2)
  # Strictly LHP poles. Conjugacy at ±k is enforced by construction/use; the
  # individual +k matrices need not be real because of tangential derivatives.
  real=-np.geomspace(1e-7,20,realcount)
  comp=[]
  for re in [1e-5,1e-4,1e-3,.01]:
   for im in np.linspace(.2,5,complexcount)*k:comp.extend([-re+1j*im,-re-1j*im])
  self.poles=np.r_[real,comp]
  xr=np.geomspace(1e-7,.08,12);yi=np.unique(np.r_[np.linspace(-5,5,121),[-np.sqrt(2),-np.sqrt(4/3),-1,0,1,np.sqrt(4/3),np.sqrt(2)]])*k
  z=np.array([x+1j*y for x in xr for y in yi])
  F=np.array([exterior(zz,k,cfg) for zz in z]);target=F-z[:,None,None]*self.A-self.B
  basis=1/(z[:,None]-self.poles[None,:]);scale=np.linalg.norm(basis,axis=0)
  # Row/column scales are fixed from target entries, not frequency-dependent.
  self.entryscale=np.maximum(np.sqrt(np.mean(abs(F)**2,axis=0)),1e-7)
  Y=(target/self.entryscale).reshape(len(z),100)
  self.coeff=np.linalg.lstsq(basis/scale,Y,rcond=1e-11)[0]/scale[:,None]
  self.coeff=self.coeff.reshape(-1,10,10)*self.entryscale
  self.train_error=float(np.max(np.linalg.norm(F-self(z),axis=(1,2))/np.maximum(np.linalg.norm(F,axis=(1,2)),1e-30)))
 def __call__(self,z):
  z=np.asarray(z);return z[...,None,None]*self.A+self.B+np.einsum('...p,pij->...ij',1/(z[...,None]-self.poles),self.coeff)

def sigma(z,k,cfg,Nfunc,details=False):
 U,D,T,_=schur(z,k,cfg);B=D[QP]-Nfunc(z)@U[QP];_,R=np.linalg.qr(U);Bi=B@np.linalg.inv(R);norms=np.linalg.norm(Bi,axis=1);_,sv,vh=np.linalg.svd(Bi/norms[:,None]);value=float(sv[-1])
 return (value,(U,D,T,B,R,vh,norms))if details else value

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--k',type=float,default=np.pi/256);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=Config(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295)
 tick=time.monotonic();f=Fit(a.k,cfg,24,20);print('fit',time.monotonic()-tick,'train',f.train_error,flush=True)
 xr=np.geomspace(1e-8,.05,25);yi=np.unique(np.r_[np.linspace(0,5,85),[1/np.sqrt(2),1,np.sqrt(4/3),np.sqrt(2)]])*a.k
 vals=np.empty((len(xr),len(yi)));errs=[]
 for i,x in enumerate(xr):
  for j,y in enumerate(yi):
   z=x+1j*y;vals[i,j]=sigma(z,a.k,cfg,f);N=exterior(z,a.k,cfg);errs.append(float(np.linalg.norm(f(z)-N)/np.linalg.norm(N)))
 seeds=[]
 for i in range(len(xr)):
  for j in range(len(yi)):
   if vals[i,j]<=np.min(vals[max(0,i-1):min(len(xr),i+2),max(0,j-1):min(len(yi),j+2)]):seeds.append((xr[i],yi[j],vals[i,j]))
 refined=[]
 for x,y,v in seeds:
  def fun(w):
   try:return sigma(10**w[0]+1j*a.k*w[1],a.k,cfg,f)
   except ValueError:return 1.
  opt=optimize.minimize(fun,[np.log10(x),y/a.k],method='Nelder-Mead',bounds=[(-9.8,np.log10(.08)),(0,5)],options={'xatol':1e-9,'fatol':1e-12,'maxiter':250})
  z=10**opt.x[0]+1j*a.k*opt.x[1]
  refined.append({'lambda':[float(z.real),float(z.imag)],'sigma_min':float(opt.fun),'lower_bound':bool(opt.x[0]<-9.79)})
 mini=np.unravel_index(np.argmin(vals),vals.shape)
 out={'scope':'Stable-pole rational full-coupled exterior DtN fit at one tangential Fourier frequency; bounded continuum frozen-coefficient screening only, not CPBC proof or finite-difference implementation.','config':asdict(cfg),'k':a.k,'poles':len(f.poles),'pole_max_real':float(f.poles.real.max()),'train_error':f.train_error,'heldout_relative_errors':{'max':max(errs),'p50':float(np.median(errs))},'grid_min':{'lambda':[float(xr[mini[0]]),float(yi[mini[1]])],'sigma_min':float(vals[mini])},'refined':refined,'seconds':time.monotonic()-tick}
 a.output.write_text(json.dumps(out,indent=2)+'\n');np.savez(a.output.with_suffix('.npz'),k=a.k,A=f.A,B=f.B,poles=f.poles,coeff=f.coeff);print(json.dumps(out),flush=True)
