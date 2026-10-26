"""Finite native lapse/shift rate data with source-complete constraint outflow.

A reflecting finite gauge control, not a general transparent boundary. Native
bulk/scri RWM equations stay unchanged; no extra evolved field or field reset.
"""
import numpy as np
from goal1_native_constraint_outflow import Model as Parent
from goal1_native_weighted_linear import mul,coefficients
class Model(Parent):
 def __init__(self,N,H=.08,constraint_frame='physical'):
  super().__init__(N,H,constraint_frame=constraint_frame,gauge='metric');size=8*self.M;start=size-8;B=self.boundary_operator.copy();B[2]=self.full_operator[start+2];B[3]=self.full_operator[start+3];E=np.zeros((size,self.dim));E[:8,:2]=self.lead
  for i in range(N):E[8*(i+1):8*(i+2),2+8*i:2+8*(i+1)]=np.eye(8)
  E[start:start+4,-4:]=np.eye(4);K=B[:,start+4:];self.dependent_boundary_condition=float(np.linalg.cond(K/np.linalg.norm(K,axis=1)[:,None]));lift=np.linalg.solve(K,np.eye(4));E[start+4:]=mul(-lift,mul(B,E));source=np.zeros((size,4));source[start+4:]=lift;self.boundary_operator=B;self.boundary_full=source;self.expand_matrix=E;self.A=mul(self.full_operator[self.selected],E);self.boundary_source=mul(self.full_operator[self.selected],source);flow=mul(self.endpoint_full_operator,E);self.compatibility_operator=flow-mul(self.lead,flow[:2]);self.compatibility_norm=float(np.linalg.svd(self.compatibility_operator,compute_uv=False)[0]);self.compatibility_SAT=np.zeros_like(self.A);self.boundary_rhs_kind=f'{constraint_frame}_constraint_native_gauge_rate'
  a,b,c=coefficients(float(H));T=b+c;P=np.block([[T[:4,:4],a[:4,4:]/H],[H*c[4:,:4],T[4:,4:]]]);vals,R=np.linalg.eig(P);R=R[:,vals.real>0].real;Bp=self.principal_boundary_rows.copy();Bp[2]=np.r_[T[2,:4],a[2,4:]/H];Bp[3]=np.r_[T[3,:4],a[3,4:]/H];K=mul(Bp,R);K/=np.linalg.norm(K,axis=1)[:,None];self.incoming_boundary_rank=int(np.linalg.matrix_rank(K,tol=1e-10));self.incoming_boundary_condition=float(np.linalg.cond(K));self.principal_boundary_rows=Bp
