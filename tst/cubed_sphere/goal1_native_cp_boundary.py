"""Source-derived incoming constraint boundary for native8 linearized Z4c.

Existing native curvature/trace/Theta/connection values are obtained from four
boundary equations; no additional field, momentum or gradient is evolved.
Full native coefficients and derivative commutators enter the constraint rows.
"""
import numpy as np
import mpmath as mp
from functools import lru_cache
from goal1_native_weighted_fitted import Model as Parent
from goal1_native_weighted_linear import mul,coefficients
import goal1_native_constraint_finite as finite
@lru_cache(None)
def constraint_coefficients(o):
 with mp.workdps(100):
  o=mp.mpf(o);C=finite.matrices(o)[1];Z0p=mp.diff(lambda z:finite.matrices(z)[1][0],o)[3,:];Z1p=mp.diff(lambda z:finite.matrices(z)[1][1],o)[3,:]
  return [np.array(A.tolist(),float)for A in C],np.array(Z0p.tolist(),float).ravel(),np.array(Z1p.tolist(),float).ravel()
class Model(Parent):
 def __init__(self,N,H=.08):
  super().__init__(N,H,eta=0);size=8*self.M;start=size-8;r=self.r[-1];L=2-H;D=self.first_operator[-8:];D2=self.second_operator[-8:];select=np.zeros((8,size));select[:,start:]=np.eye(8);C,z0p,z1p=constraint_coefficients(float(H));constraint=mul(C[0],select)+mul(C[1],D)+mul(C[2],D2);Zp=mul(z0p[None,:],select)[0]+mul((C[0][3]+z1p)[None,:],D)[0]+mul(C[1][3:4],D2)[0];B=np.zeros((4,size));B[0]=constraint[0]+2*constraint[1];B[1]=constraint[0]-2*constraint[1]+8*r*H*(D[6]-Zp)-16*r*(select[6]-constraint[3]);B[2]=2*r*H*D[2];B[2,start+5]-=L;B[3]=-r*L*H*D[0]+2*r*H*D[2]+2*r*H*D[3];B[3,start+7]+=L
  E=np.zeros((size,self.dim));E[:8,:2]=self.lead
  for i in range(N):E[8*(i+1):8*(i+2),2+8*i:2+8*(i+1)]=np.eye(8)
  E[start:start+4,-4:]=np.eye(4);K=B[:,start+4:];self.dependent_boundary_condition=float(np.linalg.cond(K/np.linalg.norm(K,axis=1)[:,None]));lift=np.linalg.solve(K,np.eye(4));E[start+4:]=mul(-lift,mul(B,E));source=np.zeros((size,4));source[start+4:]=lift;self.boundary_operator=B;self.boundary_full=source;self.expand_matrix=E;self.A=mul(self.full_operator[self.selected],E);self.boundary_source=mul(self.full_operator[self.selected],source);flow=mul(self.endpoint_full_operator,E);self.compatibility_operator=flow-mul(self.lead,flow[:2]);self.compatibility_norm=float(np.linalg.svd(self.compatibility_operator,compute_uv=False)[0]);self.compatibility_SAT=np.zeros_like(self.A);self.boundary_rhs_kind='actual_incoming_constraints_plus_gauge'
  # Highest differential-grade restriction, analysis only: (q_Omega,p_native).
  a,b,c=coefficients(float(H));T=b+c;P=np.block([[T[:4,:4],a[:4,4:]/H],[H*c[4:,:4],T[4:,4:]]]);v,V=np.linalg.eig(P);R=V[:,v.real>0].real;Hrow=np.r_[C[2][0,:4],C[1][0,4:]];Mrow=np.r_[C[2][1,:4],C[1][1,4:]];Zrow=np.r_[C[1][3,:4],C[0][3,4:]];Thetap=np.zeros(8);Thetap[6]=1;Bp=np.zeros((4,8));Bp[0]=Hrow+2*Mrow;Bp[1]=Hrow-2*Mrow+8*r*H*(Thetap-Zrow);Bp[2,2]=2*r*H;Bp[2,5]=-L;Bp[3,0]=-r*L*H;Bp[3,2:4]=2*r*H;Bp[3,7]=L;Rin=mul(Bp,R);Rin/=np.linalg.norm(Rin,axis=1)[:,None];self.incoming_boundary_rank=int(np.linalg.matrix_rank(Rin,tol=1e-10));self.incoming_boundary_condition=float(np.linalg.cond(Rin));self.principal_boundary_rows=Bp
 def data_from_jets(self,o,y,dy,ddy):
  C,z0p,z1p=constraint_coefficients(float(o));V=np.einsum('ij,j->i',C[0],y)+np.einsum('ij,j->i',C[1],dy)+np.einsum('ij,j->i',C[2],ddy);Zp=np.dot(z0p,y)+np.dot(C[0][3]+z1p,dy)+np.dot(C[1][3],ddy);r=np.sqrt(1-o);L=2-o
  return np.array([V[0]+2*V[1],V[0]-2*V[1]+8*r*o*(dy[6]-Zp)-16*r*(y[6]-V[3]),2*r*o*dy[2]-L*y[5],-r*L*o*dy[0]+2*r*o*dy[2]+2*r*o*dy[3]+L*y[7]])
