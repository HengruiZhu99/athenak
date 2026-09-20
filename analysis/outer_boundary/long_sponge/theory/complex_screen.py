"""Bounded complex RHP search for original zero_rate with kappa zero."""
import argparse,sys,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--k',type=float,nargs='+');a=p.parse_args();sys.path.insert(0,str(a.model.resolve()))
import numpy as np
from scipy import optimize
from volume import Config,QP,asdict
from boundary import assess_mode
from validate import mode
base=dict(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295,G=1,kappa=0)
out={'scope':'Bounded real/complex RHP search, no proof of stability. Includes no curvature or variable coefficients.','cases':[]};start=time.monotonic()
for name,eta,ld in [('k0_original_gauge',2,.1),('k0_weak_gauge',.02,.01)]:
 cfg=Config(**base,eta=eta,lapse_damping=ld)
 for k in (a.k if a.k is not None else [np.pi/256,np.pi/1024]):
  xr=np.geomspace(1e-8,.05,32);yi=np.unique(np.r_[np.linspace(0,5,111),[1/np.sqrt(2),1,np.sqrt(4/3),np.sqrt(2)]])*k
  vals=np.empty((len(xr),len(yi)));errors={}
  def sigma(z):
   try:return assess_mode(z,k,cfg)['sigma_min']
   except ValueError as exc:
    key=(float(z.real),float(z.imag));errors[key]=str(exc);return 1.
  for i,x in enumerate(xr):
   for j,y in enumerate(yi):vals[i,j]=sigma(x+1j*y)
  seeds=[]
  for i in range(len(xr)):
   for j in range(len(yi)):
    sub=vals[max(0,i-1):min(len(xr),i+2),max(0,j-1):min(len(yi),j+2)]
    if vals[i,j]<=np.min(sub):seeds.append((xr[i],yi[j],vals[i,j]))
  # Explicit weak-normal-shift surface/coordinate-scale seeds.
  for fac in [.5,1/np.sqrt(2),1, np.sqrt(4/3),np.sqrt(2)]:seeds.append((cfg.beta_n*k,fac*k,sigma(cfg.beta_n*k+1j*fac*k)))
  roots=[];refined=[]
  for x,y,v in seeds:
   fit=optimize.minimize(lambda z:sigma(10**z[0]+1j*k*z[1]),[np.log10(x),y/k],method='Nelder-Mead',bounds=[(-10,np.log10(.08)),(0,5)],options={'xatol':1e-10,'fatol':1e-13,'maxiter':260})
   z=10**fit.x[0]+1j*k*fit.x[1]
   rec={'lambda':[float(z.real),float(z.imag)],'sigma_min':float(fit.fun),'at_real_lower_bound':bool(fit.x[0]<-9.999),'seed':[float(x),float(y)]}
   refined.append(rec)
   if fit.fun>1e-8 or rec['at_real_lower_bound']:continue
   _,(U,D,T,B,R,vh,norms)=assess_mode(z,k,cfg,'zero_rate',True);E=B@np.linalg.inv(U[QP]);ns=np.linalg.norm(E,axis=1);left,_,vh=np.linalg.svd(E/ns[:,None]);lv=left[:,-1];rv=vh[-1].conj()
   def fun(zvec):
    zz=zvec[0]+1j*zvec[1]
    try:
     _,(U,D,T,B,R,vh,norms)=assess_mode(zz,k,cfg,'zero_rate',True);v=np.vdot(lv,(B@np.linalg.inv(U[QP]))@rv/ns)
     return [v.real,v.imag]
    except ValueError:return [1.,1.]
   sol=optimize.root(fun,[z.real,z.imag],tol=1e-10);zz=sol.x[0]+1j*sol.x[1]
   if not(zz.real>1e-9 and 0<=zz.imag<=5*k):continue
   check=mode(zz,k,cfg,'zero_rate')
   if check['boundary_scaled']>1e-9 or check['bulk_relative']>1e-9:continue
   if any(abs(zz-complex(*q['lambda_value']))<1e-7 for q in roots):continue
   roots.append(check)
  mini=np.unravel_index(np.argmin(vals),vals.shape)
  case={'name':name,'config':asdict(cfg),'k':float(k),'real_grid':[float(xr[0]),float(xr[-1]),len(xr)],'imag_grid':[float(yi[0]),float(yi[-1]),len(yi)],'grid_min':{'sigma_min':float(vals[mini]),'lambda':[float(xr[mini[0]]),float(yi[mini[1]])]},'roots':roots,'refined_minima':refined,'classification_failures':[{'lambda':list(z),'error':text}for z,text in errors.items()]}
  out['cases'].append(case);out['elapsed_seconds']=time.monotonic()-start;a.output.write_text(json.dumps(out,indent=2)+'\n');print(name,k,'roots',[(q['lambda_value'],q['boundary_scaled'])for q in roots],'gridmin',case['grid_min'],'seeds',len(seeds),'elapsed',out['elapsed_seconds'],flush=True)
