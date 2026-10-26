"""Actual native8 radial linearized Z4c/RWM semidiscretization; no extra fields.

The independent full tensor source verification of the analytic radial matrices
is reused. This is native coupled linear evolution, not a subsidiary equation
or scalar-wave proxy, and not nonlinear/mapped3D scientific acceptance.
"""
import numpy as np
import mpmath as mp
from functools import lru_cache
import goal1_native_radial_linear as equation
import goal1_native_weighted_endpoint as endpoint
import goal1_native_radial_scri as native
mul=lambda A,B:np.einsum('ij,jk->ik',A,B)
@lru_cache(None)
def coefficients(o):
 with mp.workdps(70):
  eye=mp.eye(8)
  def mat(s):
   cols=[equation.action(s,mp.mpf(o),eye[:,k])for k in range(8)]
   return np.array([[float(cols[j][i])for j in range(8)]for i in range(8)])
  a,b,c=mat(0),mat(1),mat(-1)
 return a,(b-c)/2,(b+c)/2-a
@lru_cache(None)
def endpoint_coefficients():
 with mp.workdps(80):
  eye=mp.eye(8);K=mp.matrix(8,8)
  for j in range(8):
   for i in range(8):K[i,j]=mp.diff(lambda o:equation.action(0,o,eye[:,j])[i],0)
 return np.array(K.tolist(),float),coefficients(0.)[0]+coefficients(0.)[1]+coefficients(0.)[2]

def interpolate_derivatives(nodes,x,order,scale):
 W,_,_=endpoint.reconstruction_weights(tuple(float(v/scale)for v in nodes));s=endpoint.SIGMA;g=endpoint.kernel()/np.linalg.norm(endpoint.kernel());E=np.zeros((8,41));z=x/scale
 for field in range(8):
  for power in range(1,5):
   if power>=order:E[field,8*(power-1)+field]=__import__('math').factorial(power)/__import__('math').factorial(power-order)*z**(power-order)/scale**order
  factor=np.prod([s-k for k in range(order)]);E[field,32]=factor*z**(s-order)*g[field]/scale**order
  factor=np.prod([s+1-k for k in range(order)]);E[field,33+field]=factor*z**(s+1-order)/scale**order
 return mul(E,W)

class Model:
 def __init__(self,N,H=.08,weak=True,eta=0):
  assert N>=8;self.N=N;self.H=H;self.h=H/N;self.o=np.r_[0,(np.arange(N)+.5)*self.h,H];self.r=np.sqrt(1-self.o);self.M=len(self.o);self.dim=8*N+6;self.weak=weak;self.eta=eta;size=8*self.M
  D1=np.zeros((self.M,self.M));D2=D1.copy()
  for i in range(self.M):
   ids=np.arange(6)if i<3 else(np.arange(self.M-6,self.M)if i>=self.M-3 else np.arange(i-2,i+3))
   D1[i,ids]=native.weights(self.o[ids],self.o[i],1,self.h);D2[i,ids]=native.weights(self.o[ids],self.o[i],2,self.h)
  self.scalar_D1=D1;self.scalar_D2=D2
  first=np.kron(D1,np.eye(8));second=np.kron(D2,np.eye(8));nodes=self.o[1:9];scale=nodes[-1]
  if weak:
   for i in range(1,5):
    for order,target in [(1,first),(2,second)]:
     B=interpolate_derivatives(nodes,self.o[i],order,scale);target[8*i:8*i+8]=0;target[8*i:8*i+8,8:72]=B
     for k in range(8):target[8*i:8*i+8,k]-=B[:,k::8].sum(axis=1)
  # Native leading pole compatibility gives two live metric components.
  self.lead=np.array([[1,0],[0,1],[-1,-2],[0,3],[0,0],[-3,3],[0,0],[0,0]],float)
  E=np.zeros((size,self.dim));E[:8,:2]=self.lead
  for i in range(N):E[8*(i+1):8*(i+2),2+8*i:2+8*(i+1)]=np.eye(8)
  E[-8:-4,-4:]=np.eye(4)
  m0,m1,m2=coefficients(float(H));A=m1+m2;B=m0/H;D=m2*H;principal=np.block([[A[:4,:4],B[:4,4:]],[D[4:,:4],A[4:,4:]]]);vals,left=np.linalg.eig(principal.T);assert np.max(abs(vals.imag))<1e-9;select=np.where(vals.real>0)[0];assert len(select)==4
  incoming=left[:,select].real.T;incoming/=np.linalg.norm(incoming,axis=1)[:,None];Lp=incoming[:,4:];Lq=incoming[:,:4];self.incoming=incoming;self.inner_speeds=vals.real;self.incoming_p_condition=float(np.linalg.cond(Lp));lift=np.linalg.solve(Lp,np.eye(4));qgradient=np.zeros((4,size))
  for k in range(4):qgradient[k,k::8]=D1[-1]
  E[-4:]=mul(-mul(lift,Lq),mul(qgradient,E));self.expand_matrix=E;self.inner_lift=lift
  L=np.zeros((size,size))
  for i in range(1,self.M):
   a,b,c=coefficients(float(self.o[i]));idx=slice(8*i,8*i+8);L[idx,idx]+=a/self.o[i];L[idx]+=mul(b+c,first[idx])+self.o[i]*mul(c,second[idx])
  K,L1=endpoint_coefficients();endpoint_full=np.zeros((8,size));endpoint_full[:,:8]=K
  if weak:
   W,_,_=endpoint.reconstruction_weights(tuple(float(v/scale)for v in nodes));T=mul(L1,W[:8]/scale);endpoint_full[:,8:72]+=T
   for k in range(8):endpoint_full[:,k]-=T[:,k::8].sum(axis=1)
  else:
   for k in range(8):endpoint_full[:,k::8]+=L1[:,k,None]*D1[0,None,:]
  L[:2]=endpoint_full[:2];self.endpoint_full_operator=endpoint_full
  self.selected=np.r_[0,1,np.arange(8,8*(N+1)),np.arange(8*(N+1),8*(N+1)+4)];assert len(self.selected)==self.dim
  self.first_operator=first;self.second_operator=second;self.full_operator=L;self.A=mul(L[self.selected],E)
  endpoint_flow=mul(endpoint_full,E);Rop=endpoint_flow-mul(self.lead,endpoint_flow[:2]);self.compatibility_operator=Rop;self.compatibility_norm=float(np.linalg.svd(Rop,compute_uv=False)[0]);C=Rop/self.compatibility_norm;self.compatibility_SAT=-eta/self.h*mul(C.T,C);self.A+=self.compatibility_SAT
  inflow=np.zeros((size,4));inflow[-4:]=lift;self.boundary_source=mul(L[self.selected],inflow);self.boundary_full=inflow
 def pack(self,y):return y.reshape(-1)[self.selected]
 def expand(self,z,incoming=None):
  v=np.einsum('ij,j->i',self.expand_matrix,z)
  if incoming is not None:v+=np.einsum('ij,j->i',self.boundary_full,incoming)
  return v.reshape(self.M,8)
 def rhs(self,z,incoming=None):
  f=np.einsum('ij,j->i',self.A,z)
  if incoming is not None:f+=np.einsum('ij,j->i',self.boundary_source,incoming)
  return f
 def step(self,z,dt,t,inflow=None):
  bc=lambda x:None if inflow is None else inflow(x)
  a=z+dt*self.rhs(z,bc(t));b=.75*z+.25*(a+dt*self.rhs(a,bc(t+dt)));return z/3+2/3*(b+dt*self.rhs(b,bc(t+dt/2)))
