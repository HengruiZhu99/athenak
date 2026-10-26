"""Native8 finite constraint/gauge boundary candidate on fixed RWM linearization.

The native fields remain chi,h,alpha,beta,OmegaA,P,Theta,OmegaLambda. Boundary
conditions are solved on those fields, with no extra evolved gradients/momenta.
Only the finite inner cut changes; the native bulk and known-scri closure remain
those of the immutable globally fitted parent. General nonlinear/3D use unadmitted.
"""
import numpy as np
from goal1_native_weighted_fitted import Model as Parent
from goal1_native_weighted_linear import mul,coefficients
class Model(Parent):
 def __init__(self,N,H=.08):
  super().__init__(N,H,eta=0);size=8*self.M;r=self.r[-1];L=2-H;start=8*(self.M-1);D=self.first_operator[-8:];B=np.zeros((4,size));B[0,start+6]=1
  B[1]=r*H*D[1];B[1,start+7]+=.5;B[1,start+1]-=1.5*H/r
  B[2]=2*r*H*D[2];B[2,start+5]-=L
  B[3]=-r*L*H*D[0]+2*r*H*D[2]+2*r*H*D[3];B[3,start+7]+=L
  E=np.zeros((size,self.dim));E[:8,:2]=self.lead
  for i in range(N):E[8*(i+1):8*(i+2),2+8*i:2+8*(i+1)]=np.eye(8)
  free=np.array([0,1,2,4]);dependent=np.array([3,5,6,7]);E[start+free,-4:]=np.eye(4);C=B[:,start+dependent];self.dependent_boundary_condition=float(np.linalg.cond(C));lift=np.linalg.solve(C,np.eye(4));E[start+dependent]=mul(-lift,mul(B,E));source=np.zeros((size,4));source[start+dependent]=lift
  self.boundary_operator=B;self.boundary_equation_lift=source;self.boundary_rhs_kind='nativeTheta_OmegaZcov_incomingAlpha_incomingBeta';self.free_boundary_native_fields=free;self.dependent_boundary_native_fields=dependent;self.expand_matrix=E;self.selected=np.r_[0,1,np.arange(8,8*(N+1)),start+free];self.A=mul(self.full_operator[self.selected],E);self.boundary_full=source;self.boundary_source=mul(self.full_operator[self.selected],source)
  flow=mul(self.endpoint_full_operator,E);self.compatibility_operator=flow-mul(self.lead,flow[:2]);self.compatibility_norm=float(np.linalg.svd(self.compatibility_operator,compute_uv=False)[0]);self.compatibility_SAT=np.zeros_like(self.A)
  # Analysis only: incoming trace rank in actual (q_Omega,p_native) principal frame.
  a,b,c=coefficients(float(H));T=b+c;P=np.block([[T[:4,:4],a[:4,4:]/H],[H*c[4:,:4],T[4:,4:]]]);v,V=np.linalg.eig(P);incoming=V[:,v.real>0].real;rows=np.zeros((4,8));rows[0,6]=1;rows[1,1]=r*H;rows[1,7]=.5;rows[2,2]=2*r*H;rows[2,5]=-L;rows[3,0]=-r*L*H;rows[3,2:4]=2*r*H;rows[3,7]=L;R=mul(rows,incoming);norms=np.linalg.norm(R,axis=1);self.incoming_boundary_rank=int(np.linalg.matrix_rank(R/norms[:,None],tol=1e-10));self.incoming_boundary_condition=float(np.linalg.cond(R/norms[:,None]));self.principal_boundary_rows=rows
 def data_from_jets(self,o,y,dy):
  r=np.sqrt(1-o);L=2-o
  return np.array([y[6],(y[7]+2*r*o*dy[1]-3*o*y[1]/r)/2,2*r*o*dy[2]-L*y[5],-r*L*o*dy[0]+2*r*o*dy[2]+2*r*o*dy[3]+L*y[7]])
