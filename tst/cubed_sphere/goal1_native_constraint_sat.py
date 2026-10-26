"""Native8 dynamic finite incoming SAT, retaining full native boundary fields.

No field/constraint value is reset; no new wave momenta or gradients are evolved.
The incoming lift is analysis of the actual second-order native source only.
General nonlinear/mapped3D stability or transparent boundary use is unadmitted.
"""
import numpy as np
from goal1_native_weighted_fitted import Model as Parent
from goal1_native_weighted_linear import coefficients,mul
class Model(Parent):
 def __init__(self,N,H=.08,gamma=1,variant='constraint'):
  super().__init__(N,H,eta=0);self.dim=8*N+10;size=8*self.M;r=self.r[-1];L=2-H;start=8*(self.M-1);D=self.first_operator[-8:];B=np.zeros((4,size));B[0,start+6]=1;B[1]=r*H*D[1];B[1,start+7]+=.5;B[1,start+1]-=1.5*H/r;B[2]=2*r*H*D[2];B[2,start+5]-=L;B[3]=-r*L*H*D[0]+2*r*H*D[2]+2*r*H*D[3];B[3,start+7]+=L
  a,b,c=coefficients(float(H));T=b+c;P=np.block([[T[:4,:4],a[:4,4:]/H],[H*c[4:,:4],T[4:,4:]]]);vals,V=np.linalg.eig(P);R=V[:,vals.real>0].real;R/=np.linalg.norm(R,axis=0)[None,:];lambda_plus=float(max(vals.real));Bprin=np.zeros((4,8));Bprin[0,6]=1;Bprin[1,1]=r*H;Bprin[1,7]=.5;Bprin[2,2]=2*r*H;Bprin[2,5]=-L;Bprin[3,0]=-r*L*H;Bprin[3,2:4]=2*r*H;Bprin[3,7]=L
  if variant=='principal':
   Bprin=self.incoming.copy();B=mul(Bprin[:,:4],D[:4]);B[:,start+4:start+8]+=Bprin[:,4:]
  elif variant!='constraint':raise ValueError('Unknown native boundary variant')
  K=mul(Bprin,R);lift=mul(R[4:],np.linalg.inv(K));self.incoming_condition=float(np.linalg.cond(K/np.linalg.norm(K,axis=1)[:,None]));self.incoming_rank=int(np.linalg.matrix_rank(K,tol=1e-10));self.characteristic_residual=float(np.linalg.norm(mul(P,R)-lambda_plus*R)/max(1,np.linalg.norm(P)*np.linalg.norm(R)));self.variant=variant;self.gamma=gamma
  E=np.zeros((size,self.dim));E[:8,:2]=self.lead;E[8:,2:]=np.eye(size-8);self.expand_matrix=E;self.selected=np.r_[0,1,np.arange(8,size)];penalty=np.zeros((size,4));penalty[start+4:start+8]=-gamma*lambda_plus/self.h*lift;self.boundary_operator=B;self.native_SAT_lift=penalty;self.boundary_source=-penalty[self.selected];self.boundary_full=np.zeros((size,4));self.A=mul((self.full_operator+mul(penalty,B))[self.selected],E);self.native_SAT=mul(mul(penalty[self.selected],B),E);self.boundary_rhs_kind=variant
  flow=mul(self.endpoint_full_operator,E);self.compatibility_operator=flow-mul(self.lead,flow[:2]);self.compatibility_norm=float(np.linalg.svd(self.compatibility_operator,compute_uv=False)[0]);self.compatibility_SAT=np.zeros_like(self.A)
 def data_from_jets(self,o,y,dy):
  r=np.sqrt(1-o);L=2-o
  if self.variant=='constraint':return np.array([y[6],(y[7]+2*r*o*dy[1]-3*o*y[1]/r)/2,2*r*o*dy[2]-L*y[5],-r*L*o*dy[0]+2*r*o*dy[2]+2*r*o*dy[3]+L*y[7]])
  return np.einsum('ij,j->i',self.incoming,np.r_[dy[:4],y[4:]])
