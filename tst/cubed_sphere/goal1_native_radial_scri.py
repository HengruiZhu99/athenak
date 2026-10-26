"""Conditional native spherical-sector scri prototype; never a 3D acceptance solver.

Native field equations are evaluated by the original tensor C++ implementation.
No spatial gradients or added momenta are independently integrated. Endpoint
relations are the derived smooth-branch DAE; its weak-branch validity is open.
"""
from pathlib import Path
import atexit,ctypes,hashlib,json,math
from functools import lru_cache
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'build-goal1-minkowski-20261018/native-radial-scri-v1'
ACTIVE=[0,1,2,3,4,5,6,8]
ROWS22=[0,18,7,17,19,14,1,4,8,11]
CONFIG22=[0,1,2,3,4,5,6,18,19,20,21]
class API:
 def __init__(self):
  self.path=WORK/'libnative_radial_scri.dylib';self.lib=ctypes.CDLL(str(self.path));p=ctypes.POINTER(ctypes.c_double);self.p=p
  self.lib.radial_error.restype=ctypes.c_char_p
  self.lib.radial_height_constant.restype=ctypes.c_double
  self.lib.radial_reference.argtypes=[ctypes.c_int,p,p]
  self.lib.radial_oracle.argtypes=[ctypes.c_int,p,ctypes.c_double,ctypes.c_double,ctypes.c_double,p,p]
  self.lib.radial_source.argtypes=[ctypes.c_int,p,p,p,p,p,p,p,p]
  self.lib.radial_jets.argtypes=[ctypes.c_int,p,p,p,p,p]
  self.check(self.lib.radial_init());atexit.register(self.lib.radial_finalize)
  self.centre=self.lib.radial_height_constant()+.05
 def check(self,code):
  if code:raise RuntimeError(self.lib.radial_error().decode())
 def ptr(self,a):return a.ctypes.data_as(self.p)
 def reference(self,r):
  r=np.ascontiguousarray(r,dtype=float);out=np.zeros((len(r),3,10));self.check(self.lib.radial_reference(len(r),self.ptr(r),self.ptr(out)));return out
 def oracle(self,r,t,eps,centre=None):
  r=np.ascontiguousarray(r,dtype=float);out=np.zeros((len(r),3,10));rate=np.zeros((len(r),10));self.check(self.lib.radial_oracle(len(r),self.ptr(r),t,eps,self.centre if centre is None else centre,self.ptr(out),self.ptr(rate)));return out,rate
 def source(self,r,v,d,dd,extra=None):
  r=np.ascontiguousarray(r,dtype=float);v=np.ascontiguousarray(v);d=np.ascontiguousarray(d);dd=np.ascontiguousarray(dd);extra=np.zeros_like(v)if extra is None else np.ascontiguousarray(extra);out=np.zeros_like(v);meta=np.zeros((len(r),18));end=np.zeros(22);self.check(self.lib.radial_source(len(r),*[self.ptr(a)for a in [r,v,d,dd,extra,out,meta,end]]));assert np.isfinite(out).all()and np.isfinite(meta).all();return out,meta,end
 def jets(self,r,v,d,dd):
  arr=[np.ascontiguousarray(a,dtype=float)for a in [r,v,d,dd]];out=np.zeros((len(r),22,13));self.check(self.lib.radial_jets(len(r),*[self.ptr(a)for a in arr],self.ptr(out)));return out
_api=None
def api():
 global _api
 if _api is None:_api=API()
 return _api
@lru_cache(None)
def unit_weights(z,order):
 with mp.workdps(70):
  A=mp.matrix([[mp.mpf(q)**k for q in z]for k in range(len(z))]);b=mp.matrix([mp.factorial(order)if k==order else 0 for k in range(len(z))]);w=mp.lu_solve(A,b)
 return np.array([float(x)for x in w])
def weights(x,at,order,h):return unit_weights(tuple(float(round((v-at)/h,12))for v in x),order)/h**order

def omega(r):
 r=np.asarray(r);o=1-r*r;mid=(r>.05)&(r<.95);s=(r[mid]-.05)/.9;g=-1/s+1/(1-s);e=np.exp(-abs(g));w=np.where(g<=0,e/(1+e),1/(1+e));o[mid]=1-w*r[mid]**2;o[r<=.05]=1;return o
class Model:
 def __init__(self,N,diss=.1):
  self.N=N;self.h=1/N;self.r=np.r_[ (np.arange(N)+.5)/N,1.];self.M=N+1;self.ref=api().reference(self.r);self.diss=diss
  self.edges=np.r_[0,(self.r[:-1]+self.r[1:])/2,1];self.mass=np.diff(self.edges)
  xr=np.r_[-self.r[::-1],self.r];imap=np.r_[np.arange(self.M-1,-1,-1),np.arange(self.M)];negative=np.r_[np.ones(self.M,dtype=bool),np.zeros(self.M,dtype=bool)];self.D={}
  for order in [1,2]:
   even=np.zeros((self.M,self.M));odd=even.copy()
   for i in range(self.M):
    c=self.M+i;ids=np.arange(len(xr)-6,len(xr))if i>=N-2 else np.arange(c-2,c+3);w=weights(xr[ids],self.r[i],order,self.h)
    for k,a in zip(ids,w):even[i,imap[k]]+=a;odd[i,imap[k]]+=a*(-1 if negative[k]else 1)
   even[np.arange(self.M),np.arange(self.M)]-=even.sum(axis=1)
   self.D[order]=(even,odd)
  Ae=[];Ao=[];co=[]
  for i in range(len(xr)-3):
   ids=np.arange(i,i+4);w=weights(xr[ids],xr[i],3,self.h)*self.h**3;e=np.zeros(self.M);o=e.copy()
   for k,a in zip(ids,w):e[imap[k]]+=a;o[imap[k]]+=a*(-1 if negative[k]else 1)
   Ae.append(e);Ao.append(o);co.append(float(np.mean(omega(abs(xr[ids])))**2))
  self.Q=[]
  for A in [np.array(Ae),np.array(Ao)]:self.Q.append(-diss/128*np.einsum('ki,k,kj->ij',A,np.array(co),A)/self.mass[:,None])
  self.quad_weights=[]
  for q in [8,16]:
   with mp.workdps(40):nodes,w=mp.gauss_quadrature(q,'legendre')
   nodes=np.array([float(a)for a in nodes]);w=np.array([float(a)for a in w]);lo=self.edges[:-1];hi=self.edges[1:];rr=((lo+hi)[:,None]+(hi-lo)[:,None]*nodes)/2;ref=api().reference(rr.ravel())[:,0,:].reshape(self.M,q,10);vol=4*math.pi*np.einsum('ij,j->i',rr*rr*np.sqrt(ref[:,:,6]/ref[:,:,0]),w)*(hi-lo)/2;self.quad_weights.append(vol)
  self.volume=self.quad_weights[-1];self.volume_uncertainty=float(np.max(abs(self.quad_weights[0]-self.volume))/self.volume.sum())
 def expand(self,y):
  if not np.isfinite(y).all()or np.min(y[:,0])<=0 or np.min(y[:,6])<=0 or np.min(y[:,1])<=0:raise RuntimeError('nonpositive/nonfinite native spherical state')
  v=np.zeros((self.M,10));v[:,ACTIVE]=y;v[:,7]=v[:,6]**-.5;v[:,9]=-.5*v[:,7]/v[:,6]*v[:,8];return v
 def apply(self,op,data):
  out=np.einsum('ij,jk->ik',op[0],data);out[:,[4,5]]=np.einsum('ij,jk->ik',op[1],data[:,[4,5]]);return out
 def derivatives(self,v):return tuple(self.ref[:,k,:]+self.apply(self.D[k],v-self.ref[:,0,:])for k in [1,2])
 def enforce(self,y):
  y=y.copy();v=self.expand(y);chi=v[-1,0];a=v[-1,6];brr=a/chi;btt=v[-1,7]/chi
  y[-1,1]=2*btt/math.sqrt(brr);y[-1,2]=-6/math.sqrt(brr);y[-1,3]=0;y[-1,4]=-2*btt/brr
  d,_=self.derivatives(v);br=(d[-1,6]*chi-a*d[-1,0])/chi**2;bt=(d[-1,7]*chi-v[-1,7]*d[-1,0])/chi**2
  y[-1,7]=chi*math.sqrt(brr)/3*(br/brr+bt/btt)
  y[-1,5]=d[-1,6]/(2*a**2)-d[-1,7]/(a*v[-1,7])+2*(1/v[-1,7]-1/a)
  return y
 def endpoint_rates(self,v,d,rates):
  rates=rates.copy();chi=v[-1,0];a=v[-1,6];gt=v[-1,7];c=rates[-1,0];at=rates[-1,6];gtt=-.5*gt/a*at
  br=a/chi;bt=gt/chi;brt=(at*chi-a*c)/chi**2;btt=(gtt*chi-gt*c)/chi**2
  grr_r=d[-1,6];gt_r=d[-1,7];chi_r=d[-1,0];brr_r=(grr_r*chi-a*chi_r)/chi**2;btt_r=(gt_r*chi-gt*chi_r)/chi**2
  adr=self.apply(self.D[1],rates)[-1];allgtt=-.5*v[:,7]/v[:,6]*rates[:,6];gtdr=np.einsum('j,j->',self.D[1][0][-1],allgtt)
  brr_rt=(adr[6]*chi+grr_r*c-at*chi_r-a*adr[0])/chi**2-2*brr_r*c/chi
  btt_rt=(gtdr*chi+gt_r*c-gtt*chi_r-gt*adr[0])/chi**2-2*btt_r*c/chi
  W=brr_r/br+btt_r/bt;Wt=brr_rt/br-brr_r*brt/br**2+btt_rt/bt-btt_r*btt/bt**2
  rates[-1,8]=chi*math.sqrt(br)/3*(Wt+W*(c/chi+.5*brt/br))
  rates[-1,2]=3*brt/br**1.5;rates[-1,1]=v[-1,1]*(btt/bt-.5*brt/br);rates[-1,4]=v[-1,4]*(btt/bt-brt/br);rates[-1,3]=0
  rates[-1,5]=adr[6]/(2*a**2)-grr_r*at/a**3-gtdr/(a*gt)+gt_r*at/(a**2*gt)+gt_r*gtt/(a*gt**2)+2*(-gtt/gt**2+at/a**2)
  rates[-1,7]=gtt;rates[-1,9]=-.5*((gtt/a-gt*at/a**2)*v[-1,8]+gt/a*rates[-1,8]);return rates
 def rhs(self,y,detail=False):
  y=self.enforce(y);v=self.expand(y);d,dd=self.derivatives(v);extra=self.apply(self.Q,v-self.ref[:,0,:]);rates,meta,end=api().source(self.r,v,d,dd,extra);native_end=rates[-1].copy();rates=self.endpoint_rates(v,d,rates)
  if detail:return rates[:,ACTIVE],{'y':y,'v':v,'d':d,'dd':dd,'rates':rates,'meta':meta,'endpoint_full_native_finite_part':end,'endpoint_native_DAE_rate_defect':rates[-1]-native_end}
  return rates[:,ACTIVE]
 def initial(self,epsilon):return self.enforce(api().oracle(self.r,0.,epsilon)[0][:,0,ACTIVE])
 def step(self,y,dt):
  a=self.enforce(y+dt*self.rhs(y));b=self.enforce(.75*y+.25*(a+dt*self.rhs(a)));return self.enforce(y/3+2/3*(b+dt*self.rhs(b)))
 def masks(self):return {'all':np.ones(self.M,bool),'interior':np.arange(self.M)<self.N,'origin':self.r<.15,'transition':(self.r>.05)&(self.r<.95),'outer':self.r>=.95,'collar6':np.arange(self.M)>=self.M-6,'scri':np.arange(self.M)==self.N}
 def norm(self,z,mask):
  q=np.sum(z[mask]**2,axis=tuple(range(1,z.ndim)))if z.ndim>1 else z[mask]**2;w=self.volume[mask];return {'rms':float(np.sqrt(np.einsum('i,i->',w,q)/w.sum())),'linf':float(np.sqrt(q.max()))}
 def report(self,y,t,epsilon):
  _,data=self.rhs(y,True);exact,erate=api().oracle(self.r,t,epsilon);obs=api().jets(self.r,data['v'],data['d'],data['dd']);target=api().jets(self.r,exact[:,0,:],exact[:,1,:],exact[:,2,:]);A=4*abs(epsilon)/.04;scale=np.ones(22)if not epsilon else np.full(22,A/.2)
  if epsilon:scale[CONFIG22]=A
  diff=(obs-target)/scale[None,:,None];result={'time':t,'N':self.N,'epsilon':epsilon,'amplitude_scale':A,'length':.2,'volume_uncertainty':self.volume_uncertainty,'errors':{},'constraints':{},'endpoint_native_DAE_rate_defect':data['endpoint_native_DAE_rate_defect'].tolist(),'endpoint_compatibility':{},'positive':bool(min(data['v'][:,[0,1,6,7]].min(),1)>0),'radial_characteristic_minimum':data['meta'][:,14].tolist(),'scri_signed_speeds':data['meta'][-1,14:16].tolist()}
  for region,mask in self.masks().items():
   result['errors'][region]={k:self.norm(z,mask)for k,z in [('fields',diff[:,:,0]),('first',.2*diff[:,:,1:4]),('configuration-second',.04*diff[:,CONFIG22,4:])]}
   result['constraints'][region]={name:self.norm(data['meta'][:,col],mask)for name,col in [('Hraw',1),('Mconformal',2),('Mphysical',3),('Zconformal',4),('Zphysical',5),('Theta',6),('det-minus1',7),('Atrace',8),('differential-det',9),('nullness',10)]}
   if epsilon:result['constraints'][region]['normalized']={name:{kind:val/(A/.04 if name in ['Hraw','Mconformal','Mphysical']else A/.2)for kind,val in result['constraints'][region][name].items()}for name in ['Hraw','Mconformal','Mphysical','Zconformal','Zphysical','Theta']}
  v,d=data['v'][-1],data['d'][-1];chi=v[0];a=v[6];gt=v[7];br=a/chi;bt=gt/chi;bp=(d[6]*chi-a*d[0])/chi**2;tp=(d[7]*chi-gt*d[0])/chi**2;w=2*v[4]/v[1];wp=2*v[4]/v[1]+2*d[4]/v[1]-2*v[4]*d[1]/v[1]**2;Gprime=8*chi/a+4*(d[0]/a-chi*d[6]/a**2);ell=(-6+bp/br-2*tp/bt)/br;result['endpoint_compatibility']={'pole_max':float(data['meta'][-1,11]),'pole_tangent_max':float(data['meta'][-1,12]),'N1':float((Gprime-2*w*wp)/-2),'tau':float(d[3]/-2),'Q_plus_ell_over_w':float((d[2]+2*d[3]-3*wp)/-2+ell/w),'native_A_rr':float(v[8]),'native_Lambda_r':float(v[5])}
  ids=np.arange(self.N-6,self.N);wn=weights(self.r[ids],1.,0,self.h);o=data['meta'][ids,0];vals=np.column_stack((data['meta'][ids,1]/o**2,data['meta'][ids,16]/o,data['meta'][ids,6]/o**2,data['meta'][ids,17]/o));result['endpoint_extrapolated_natural_constraints_H_M_Theta_Z']=np.einsum('i,ij->j',wn,vals).tolist();result['endpoint_constraint_limits_are_numerical_estimates']=True
  result['entry_screen']={'field_first_Hessian_pass':all(z['rms']<=.01 and z['linf']<=.05 for region in result['errors'].values()for z in region.values()),'positive':result['positive']}
  if epsilon:result['entry_screen']['constraint_pass']=all(z['rms']<=.01 and z['linf']<=.05 for region in result['constraints'].values()for z in region['normalized'].values())
  else:result['entry_screen']['stationary_field_pass']=max(region['fields']['linf']for region in result['errors'].values())<=1e-9;result['entry_screen']['stationary_raw_constraint_pass']=max(region[k]['linf']for region in result['constraints'].values()for k in ['Hraw','Mconformal','Mphysical','Zconformal','Zphysical','Theta'])<=1e-8
  return result,data
