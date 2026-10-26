"""Finite physical-constraint outflow on the actual native8 linearization.

Constraint/gauge observables are stateless boundary functions of existing native
fields. Their derivatives use full source/geometry commutators; no additional
Z4, momentum, configuration gradient or coordinate potential is integrated.
"""
import mpmath as mp
import numpy as np
from functools import lru_cache
from goal1_native_cp_metric_boundary import Model as MetricParent
from goal1_native_weighted_linear import mul,coefficients
from goal1_native_cp_boundary import constraint_coefficients
import goal1_native_constraint_finite as finite

def frame(o,kind):
 r=mp.sqrt(1-o);L=2-o
 if kind=='physical':return mp.matrix([[-L,-2*r],[2*r,L]])/o
 if kind=='natural':return mp.eye(2)/o**2
 if kind=='native':return mp.eye(2)
 raise ValueError('Unknown constraint boundary frame')
@lru_cache(None)
def constraint_operators(o,kind):
 with mp.workdps(100):
  o=mp.mpf(o);K,res=finite.closure(o);assert res<mp.mpf('1e-70');assert mp.norm(K[2],p=mp.inf)<mp.mpf('1e-70');r=mp.sqrt(1-o);L=2-o;a=(L+2*r)/L;b=2*r*o*o/L;rho=o/r;S=frame(o,kind);Sp=mp.diff(lambda z:frame(z,kind),o);A=a*S*K[0][2:4,:2];B=a*S*K[1][2:4,2:4]+b*S;C=a*S*K[0][2:4,2:4]+b*Sp-rho*S
  assert mp.norm(K[1][2:4,:2],p=mp.inf)<mp.mpf('1e-70');return [np.array(M.tolist(),float)for M in [A,B,C]]
class Model(MetricParent):
 def __init__(self,N,H=.08,constraint_frame='physical',gauge='metric'):
  super().__init__(N,H);size=8*self.M;start=size-8;select=np.zeros((8,size));select[:,start:]=np.eye(8);D=self.first_operator[-8:];D2=self.second_operator[-8:];C,z0p,z1p=constraint_coefficients(float(H));V=mul(C[0],select)+mul(C[1],D)+mul(C[2],D2);Zp=mul(z0p[None,:],select)[0]+mul((C[0][3]+z1p)[None,:],D)[0]+mul(C[1][3:4],D2)[0];Qp=np.vstack((D[6],Zp));Aq,Bq,Cq=constraint_operators(float(H),constraint_frame);B=self.boundary_operator.copy();B[:2]=mul(Aq,V[:2])+mul(Bq,Qp)+mul(Cq,V[2:4]);r=self.r[-1];L=2-H
  if gauge=='principal':
   B[2]=2*r*H*D[2];B[2,start+5]-=L;B[3]=-r*L*H*D[0]+2*r*H*D[2]+2*r*H*D[3];B[3,start+7]+=L
  elif gauge!='metric':raise ValueError('Unknown gauge boundary')
  E=np.zeros((size,self.dim));E[:8,:2]=self.lead
  for i in range(N):E[8*(i+1):8*(i+2),2+8*i:2+8*(i+1)]=np.eye(8)
  E[start:start+4,-4:]=np.eye(4);K=B[:,start+4:];self.dependent_boundary_condition=float(np.linalg.cond(K/np.linalg.norm(K,axis=1)[:,None]));lift=np.linalg.solve(K,np.eye(4));E[start+4:]=mul(-lift,mul(B,E));source=np.zeros((size,4));source[start+4:]=lift;self.boundary_operator=B;self.boundary_full=source;self.expand_matrix=E;self.A=mul(self.full_operator[self.selected],E);self.boundary_source=mul(self.full_operator[self.selected],source);flow=mul(self.endpoint_full_operator,E);self.compatibility_operator=flow-mul(self.lead,flow[:2]);self.compatibility_norm=float(np.linalg.svd(self.compatibility_operator,compute_uv=False)[0]);self.compatibility_SAT=np.zeros_like(self.A);self.constraint_frame=constraint_frame;self.gauge_boundary=gauge;self.constraint_boundary_operators=(Aq,Bq,Cq);self.boundary_rhs_kind=f'{constraint_frame}_constraint_outflow_{gauge}_gauge'
  a,b,c=coefficients(float(H));T=b+c;P=np.block([[T[:4,:4],a[:4,4:]/H],[H*c[4:,:4],T[4:,4:]]]);vals,R=np.linalg.eig(P);R=R[:,vals.real>0].real;Hp=np.r_[C[2][0,:4],C[1][0,4:]];Mp=np.r_[C[2][1,:4],C[1][1,4:]];Zprime=np.r_[C[1][3,:4],C[0][3,4:]];Thetap=np.zeros(8);Thetap[6]=1;Bp=self.principal_boundary_rows.copy();Bp[:2]=mul(Aq,np.vstack((Hp,Mp)))+mul(Bq,np.vstack((Thetap,Zprime)))
  if gauge=='principal':
   Bp[2]=0;Bp[2,2]=2*r*H;Bp[2,5]=-L;Bp[3]=0;Bp[3,0]=-r*L*H;Bp[3,2:4]=2*r*H;Bp[3,7]=L
  Rb=mul(Bp,R);Rb/=np.linalg.norm(Rb,axis=1)[:,None];self.incoming_boundary_rank=int(np.linalg.matrix_rank(Rb,tol=1e-10));self.incoming_boundary_condition=float(np.linalg.cond(Rb));self.principal_boundary_rows=Bp
 def data_from_jets(self,o,y,dy,ddy):
  data=super().data_from_jets(o,y,dy,ddy);C,z0p,z1p=constraint_coefficients(float(o));V=np.einsum('ij,j->i',C[0],y)+np.einsum('ij,j->i',C[1],dy)+np.einsum('ij,j->i',C[2],ddy);Zp=np.einsum('i,i->',z0p,y)+np.einsum('i,i->',C[0][3]+z1p,dy)+np.einsum('i,i->',C[1][3],ddy);A,B,D=self.constraint_boundary_operators;data[:2]=np.einsum('ij,j->i',A,V[:2])+np.einsum('ij,j->i',B,np.array([dy[6],Zp]))+np.einsum('ij,j->i',D,V[2:4])
  if self.gauge_boundary=='principal':
   r=np.sqrt(1-o);L=2-o;data[2]=2*r*o*dy[2]-L*y[5];data[3]=-r*L*o*dy[0]+2*r*o*dy[2]+2*r*o*dy[3]+L*y[7]
  return data
