"""Full local frozen Jacobian around trumpet values and actual FD background jets.

This includes lower-order/background-gradient couplings. It is neither a
principal-symbol theorem nor a global variable-coefficient eigenproblem.
"""
from pathlib import Path
import json,numpy as np
from point_operator import point_rhs
from principal_symbol import symbols,TF,pairs,operator
out=Path(__file__).resolve().parent

def trumpet(x):
 x=np.asarray(x);r=np.linalg.norm(x);assert r>0;R=1+r;n=x/r;v=np.zeros(25);v[0]=(r/R)**2;v[[1,4,6]]=1;v[7]=1/R**2;v[18]=r/R;v[19:22]=x/R**2;A=(2*np.eye(3)/3-2*np.outer(n,n))/R**2;A-=np.eye(3)*np.trace(A)/3
 for k,(i,j)in enumerate(pairs):v[8+k]=A[i,j]
 return v

def background_jets(x,h):
 cache={}
 def at(s):
  key=tuple(s)
  if key not in cache:cache[key]=trumpet(np.asarray(x)+h*np.array(s))
  return cache[key]
 v=at((0,0,0));d=np.zeros((3,25));dd=np.zeros((3,3,25));up=np.zeros((3,25));w={-3:-1/60,-2:.15,-1:-.75,1:.75,2:-.15,3:1/60}
 for a in range(3):
  def q(k):
   s=[0]*3;s[a]=k;return at(s)
  d[a]=((q(3)-q(-3))/60+3*(q(-2)-q(2))/20+3*(q(1)-q(-1))/4)/h
  dd[a,a]=((q(-3)+q(3))/90-3*(q(-2)+q(2))/20+1.5*(q(-1)+q(1))-49*v/18)/h**2
  dl=q(-4)/60-2*q(-3)/15+q(-2)/2-4*q(-1)/3+7*v/12+2*q(1)/5-q(2)/30
  dr=-q(4)/60+2*q(3)/15-q(2)/2+4*q(1)/3-7*v/12-2*q(-1)/5+q(-2)/30
  up[a]=(dl if v[19+a]<0 else dr)/h
  for b in range(a+1,3):
   for i,wi in w.items():
    for j,wj in w.items():
     s=[0]*3;s[a]=i;s[b]=j;dd[a,b]+=wi*wj*at(s)/h**2
   dd[b,a]=dd[a,b]
 return v,d,dd,up

def basis_projector(v):
 B=np.zeros((25,20));B[0,0]=1;B[7,6]=1;B[17,15]=1;B[18,16]=1;B[14:17,12:15]=np.eye(3);B[19:22,17:20]=np.eye(3);A=np.zeros((3,3))
 for k,(i,j)in enumerate(pairs):A[i,j]=A[j,i]=v[8+k]
 Ah=np.einsum('ij,qij->q',A,TF)
 for q,(i,j)in enumerate(pairs):B[1+q,1:6]=TF[:,i,j];B[8+q,7:12]=TF[:,i,j];B[8+q,1:6]+=np.eye(3)[i,j]*Ah/3
 P=np.eye(25)
 for col in range(25):
  vec=P[:,col].copy();g=np.zeros((3,3));a=np.zeros((3,3))
  for q,(i,j)in enumerate(pairs):g[i,j]=g[j,i]=vec[1+q];a[i,j]=a[j,i]=vec[8+q]
  g-=np.eye(3)*np.trace(g)/3;a-=np.eye(3)*(np.trace(a)-np.sum(A*g))/3
  for q,(i,j)in enumerate(pairs):P[1+q,col]=g[i,j];P[8+q,col]=a[i,j]
 C=np.linalg.pinv(B);assert np.max(abs(C@P@B-np.eye(20)))<1e-13;assert np.max(abs(P@P-P))<1e-13
 return B,C@P

class Frozen:
 def __init__(self,x,h=.25,kappa=.1,shift=2,eta=2,principal=False):
  self.h=h;self.shift=shift;self.eta=eta;v,d,dd,up=background_jets(x,h)
  if principal:v[7:18]=0;d[:]=0;dd[:]=0;up[:]=0
  self.v=v;self.beta=v[19:22];self.principal=principal
  B,CP=basis_projector(v);N=350;eps=1e-20;V=np.repeat(v[None,:],N,axis=0).astype(complex);D=np.repeat(d[None,:,:],N,axis=0).astype(complex);DD=np.repeat(dd[None,:,:,:],N,axis=0).astype(complex);ADV=np.repeat(np.sum(self.beta[:,None]*up,axis=0)[None,:],N,axis=0).astype(complex);off=0
  for j in range(25):V[off+j,j]+=1j*eps
  off+=25
  for a in range(3):
   for j in range(25):D[off+25*a+j,a,j]+=1j*eps
  off+=75
  for a in range(3):
   for b in range(3):
    for j in range(25):DD[off+25*(3*a+b)+j,a,b,j]+=1j*eps
  off+=225
  for j in range(25):ADV[off+j,j]+=1j*eps
  F=point_rhs(V,D,DD,ADV,kappa=kappa);F[:,18:22]=0;J=F.imag.T/eps
  self.Jv=CP@J[:,:25];self.Jd=np.array([CP@J[:,25+25*a:25+25*(a+1)]for a in range(3)]);self.Jdd=np.array([[CP@J[:,100+25*(3*a+b):100+25*(3*a+b+1)]for b in range(3)]for a in range(3)]);self.Jadv=CP@J[:,325:350];self.B=B
  self.M0=self.Jv@B
  for a in range(3):self.M0+=np.outer(self.Jadv@up[a],B[19+a])
  self.Md=np.array([j@B for j in self.Jd]);self.Madv=self.Jadv@B
 def evaluate(self,xi,scheme='standard',diss=.5):
  xi=np.atleast_2d(xi);D,D2,S=symbols(xi,self.h);L=D2.sum(-1);S1=D[..., :,None]*D[...,None,:];N=len(xi)
  A=np.repeat(self.M0[None,:,:],N,axis=0).astype(complex)+np.einsum('na,aij->nij',D,self.Md)
  ddsyms=np.repeat(S[:,:,:,None],25,axis=-1)
  if scheme=='compatible':
   H=S1+np.eye(3)*(L-np.sum(D*D,axis=-1))[:,None,None]/3
   ddsyms[:,:,:,0]=H;ddsyms[:,:,:,18]=H
   ddsyms[:,:,:,19:22]=S1[:,:,:,None]
  for a in range(3):
   for b in range(3):A+=np.einsum('if,nf,fj->nij',self.Jdd[a,b],ddsyms[:,a,b],self.B)
  if scheme=='compatible':
   # Retain vector Laplacian L while changing only grad-div to D_i D_j.
   for a in range(3):A[:,12+a,17+a]+=L-np.sum(D*D,axis=-1)
  dl=sum(w*np.exp(1j*xi*k)for k,w in[(-4,1/60),(-3,-2/15),(-2,.5),(-1,-4/3),(0,7/12),(1,2/5),(2,-1/30)])/self.h
  dr=sum(w*np.exp(1j*xi*k)for k,w in[(4,-1/60),(3,2/15),(2,-.5),(1,4/3),(0,-7/12),(-1,-2/5),(-2,1/30)])/self.h
  adv=np.sum(self.beta*np.where(self.beta<0,dl,dr),axis=-1);A+=adv[:,None,None]*self.Madv
  A[:,16,6]=-2*self.v[18]
  A[:,16,16]+=adv
  for a in range(3):A[:,17+a,12+a]=self.shift;A[:,17+a,17+a]+=adv-self.eta
  ko=-diss*np.sum(np.sin(xi/2)**8,axis=-1)/self.h;A+=ko[:,None,None]*np.eye(20)
  return A

if __name__=='__main__':
 results=[];validation=[];grid=np.linspace(0,np.pi,13);xi=np.stack(np.meshgrid(grid,grid,grid,indexing='ij'),-1).reshape(-1,3);high=xi.max(-1)>=np.pi/2
 for x in [[.125]*3,[.375,.125,.125],[.625,.125,.125],[1.125,.125,.125]]:
  f=Frozen(x,kappa=0,eta=0,principal=True)
  for scheme in ['standard','compatible']:
   phases=np.array([[.23,.91,2.5],[np.pi,np.pi,np.pi]]);A=f.evaluate(phases,scheme,diss=.5);B=operator(phases,alpha=f.v[18],chi=f.v[0],beta=f.beta,shift=2,scheme=scheme,advection='upwind',diss=.5);error=float(np.max(abs(A-B)));assert error<2e-11;validation.append({'x':x,'scheme':scheme,'principal_matrix_error':error})
  for h in [.25,.125,.0625]:
   f=Frozen(x,h=h)
   for scheme in ['standard','compatible']:
    A=f.evaluate(xi,scheme);e=np.linalg.eigvals(A);growth=e.real.max(-1);ix=int(np.argmax(growth));ih=int(np.argmax(np.where(high,growth,-np.inf)));z=e*(.15*h);gain=abs(1+z+z*z/2+z*z*z/6);results.append({'x':x,'r':float(np.linalg.norm(x)),'h':h,'scheme':scheme,'max_real':float(growth[ix]),'xi_max_over_pi':(xi[ix]/np.pi).tolist(),'max_high_frequency_real':float(growth[ih]),'xi_high_max_over_pi':(xi[ih]/np.pi).tolist(),'RK3_max_abs_dt_015h':float(gain.max())});print(results[-1],flush=True)
 (out/'frozen-full-scan.json').write_text(json.dumps({'scope':'Full local Jacobian includes finite-difference background gradients/algebraic couplings, G2/eta2/kappa.1, upwind/KO.5, tangent projection. Not a global mode or principal-symbol theorem. Compatible replacement changes perturbation jets while holding baseline background jets fixed.','validation':validation,'results':results},indent=2))
