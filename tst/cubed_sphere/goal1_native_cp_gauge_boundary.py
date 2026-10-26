"""Native8 incoming constraint plus physical inward-absorbing gauge boundary.

No coordinate potential, wave momentum, independent gradient or memory is evolved.
All boundary conditions are functions of existing native fields/direct derivatives
and the unchanged native RWM source. This isolated linearized candidate is not a
nonlinear/mapped3D endpoint admission.
"""
import numpy as np
from goal1_native_cp_boundary import Model as Parent
from goal1_native_weighted_linear import mul,coefficients
import goal1_native_gauge_boundary as physical
class Model(Parent):
 def __init__(self,N,H=.08):
  super().__init__(N,H);size=8*self.M;start=size-8;D=self.first_operator[-8:];D2=self.second_operator[-8:];select=np.zeros((8,size));select[:,start:]=np.eye(8);BT,BS=physical.operators(float(H));G=[[np.array(A.tolist(),float).ravel()for A in row]for row in [BT,BS]];B=self.boundary_operator.copy()
  for k,ops in enumerate(G):B[2+k]=np.einsum('i,ij->j',ops[0],select)+np.einsum('i,ij->j',ops[1],D)+np.einsum('i,ij->j',ops[2],D2)
  E=np.zeros((size,self.dim));E[:8,:2]=self.lead
  for i in range(N):E[8*(i+1):8*(i+2),2+8*i:2+8*(i+1)]=np.eye(8)
  E[start:start+4,-4:]=np.eye(4);K=B[:,start+4:];self.dependent_boundary_condition=float(np.linalg.cond(K/np.linalg.norm(K,axis=1)[:,None]));lift=np.linalg.solve(K,np.eye(4));E[start+4:]=mul(-lift,mul(B,E));source=np.zeros((size,4));source[start+4:]=lift;self.boundary_operator=B;self.boundary_full=source;self.expand_matrix=E;self.A=mul(self.full_operator[self.selected],E);self.boundary_source=mul(self.full_operator[self.selected],source);flow=mul(self.endpoint_full_operator,E);self.compatibility_operator=flow-mul(self.lead,flow[:2]);self.compatibility_norm=float(np.linalg.svd(self.compatibility_operator,compute_uv=False)[0]);self.compatibility_SAT=np.zeros_like(self.A);self.physical_gauge_operators=G;self.boundary_rhs_kind='incoming_constraints_physical_inward_gauge'
  a,b,c=coefficients(float(H));T=b+c;P=np.block([[T[:4,:4],a[:4,4:]/H],[H*c[4:,:4],T[4:,4:]]]);vals,R=np.linalg.eig(P);R=R[:,vals.real>0].real;Bp=self.principal_boundary_rows.copy();Bp[2]=np.r_[G[0][1][:4],G[0][0][4:]];Bp[3]=np.r_[G[1][2][:4],G[1][1][4:]];K=mul(Bp,R);K/=np.linalg.norm(K,axis=1)[:,None];self.incoming_boundary_rank=int(np.linalg.matrix_rank(K,tol=1e-10));self.incoming_boundary_condition=float(np.linalg.cond(K));self.principal_boundary_rows=Bp
 def data_from_jets(self,o,y,dy,ddy):
  data=super().data_from_jets(o,y,dy,ddy)
  for k,G in enumerate(self.physical_gauge_operators):data[2+k]=np.dot(G[0],y)+np.dot(G[1],dy)+np.dot(G[2],ddy)
  return data
