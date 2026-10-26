"""Native8 globally fitted direct D1/D2 candidate, without extra evolved fields.

A stateless rank9 correction preserves the actual leading indicial cancellation
also outside the first few cells. Source, gauge and native Cauchy degrees remain
unchanged. This is not a general nonlinear/mapped3D closure admission.
"""
import numpy as np
from goal1_native_weighted_linear import Model as Parent,coefficients,endpoint_coefficients,mul
import goal1_native_weighted_endpoint as endpoint
class Model(Parent):
 def __init__(self,N,H=.08,eta=8):
  super().__init__(N,H,weak=True,eta=0);size=8*self.M;nodes=self.o[1:9];scale=nodes[-1];s=endpoint.SIGMA;g=endpoint.kernel()/np.linalg.norm(endpoint.kernel());W,_,_=endpoint.reconstruction_weights(tuple(float(v/scale)for v in nodes));extract=np.zeros((9,size));extract[:,8:72]=W[32:]
  for k in range(8):extract[:,k]-=W[32:,k::8].sum(axis=1)
  phi=np.zeros((size,9));exact1=phi.copy();exact2=phi.copy()
  for i,o in enumerate(self.o[1:],1):
   x=o/scale
   for field in range(8):
    phi[8*i+field,0]=x**s*g[field];phi[8*i+field,1+field]=x**(s+1)
    exact1[8*i+field,0]=s*x**(s-1)*g[field]/scale;exact1[8*i+field,1+field]=(s+1)*x**s/scale
    exact2[8*i+field,0]=s*(s-1)*x**(s-2)*g[field]/scale**2;exact2[8*i+field,1+field]=(s+1)*s*x**(s-1)/scale**2
  first=np.kron(self.scalar_D1,np.eye(8));second=np.kron(self.scalar_D2,np.eye(8))
  for base,exact in [(first,exact1),(second,exact2)]:
   delta=exact-mul(base,phi);delta[:8]=0; base+=mul(delta,extract)
  self.first_operator=first;self.second_operator=second;self.weak_extractor=extract;self.weak_basis=phi
  # Rebuild finiteinner coupling with actual corrected native q derivatives.
  E=np.zeros((size,self.dim));E[:8,:2]=self.lead
  for i in range(N):E[8*(i+1):8*(i+2),2+8*i:2+8*(i+1)]=np.eye(8)
  E[-8:-4,-4:]=np.eye(4);qgradient=first[-8:-4];assert abs(qgradient[:,-4:]).max()==0
  E[-4:]=mul(-mul(self.inner_lift,self.incoming[:,:4]),mul(qgradient,E));self.expand_matrix=E
  L=np.zeros((size,size))
  for i,o in enumerate(self.o[1:],1):
   a,b,c=coefficients(float(o));ix=slice(8*i,8*i+8);L[ix,ix]+=a/o;L[ix]+=mul(b+c,first[ix])+o*mul(c,second[ix])
  K,L1=endpoint_coefficients();F0=np.zeros((8,size));F0[:,:8]=K;T=mul(L1,W[:8]/scale);F0[:,8:72]+=T
  for k in range(8):F0[:,k]-=T[:,k::8].sum(axis=1)
  L[:2]=F0[:2];self.endpoint_full_operator=F0;self.full_operator=L;self.A=mul(L[self.selected],E);R=mul(F0,E);self.compatibility_operator=R-mul(self.lead,R[:2]);self.compatibility_norm=float(np.linalg.svd(self.compatibility_operator,compute_uv=False)[0]);C=self.compatibility_operator/self.compatibility_norm;self.compatibility_SAT=-eta/self.h*mul(C.T,C);self.A+=self.compatibility_SAT;self.eta=eta
  lift=np.zeros((size,4));lift[-4:]=self.inner_lift;self.boundary_full=lift;self.boundary_source=mul(L[self.selected],lift)
