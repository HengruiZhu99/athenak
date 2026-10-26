"""Analysis-only native endpoint algebraic boundary discretization.

Six actual source time-compatibility equations are imposed on stateless regular
first jets of existing native fields. No new Cauchy fields, gradient, wave momentum,
constraint reset or leading reference-value condition is introduced.
"""
import numpy as np
from goal1_native_constraint_outflow import Model as Parent
from goal1_native_weighted_linear import mul
class Model(Parent):
 def __init__(self,N,H=.08):
  super().__init__(N,H,constraint_frame='physical',gauge='metric');R=self.compatibility_operator;rows=np.array([2,3,4,5,6,7]);C=R[rows];block=C[:,2:10].copy();scale=np.linalg.norm(block,axis=1);work=block/scale[:,None];pivots=[]
  for _ in range(6):
   norms=np.linalg.norm(work,axis=0);j=int(np.argmax(norms));pivots.append(j);v=work[:,j]/norms[j];work-=np.einsum('i,j->ij',v,np.einsum('i,ij->j',v,work))
  dep=2+np.array(pivots);self.dependent_native_slots=pivots;keep=np.array([i for i in range(self.dim)if i not in dep]);T=np.zeros((self.dim,len(keep)));T[keep,np.arange(len(keep))]=1;K=C[:,dep];row=np.linalg.norm(K,axis=1);self.endpoint_solve_condition=float(np.linalg.cond(K/row[:,None]));T[dep]=np.linalg.solve(K,-mul(C,T));self.parent_A=self.A.copy();self.parent_expand=self.expand_matrix.copy();self.parent_selected=self.selected.copy();self.dae_transform=T;self.dae_keep=keep;self.dae_dependent=dep;self.A=mul(self.A[keep],T);self.boundary_source=self.boundary_source[keep];self.expand_matrix=mul(self.expand_matrix,T);self.selected=self.selected[keep];self.dim=len(keep);self.compatibility_operator=mul(R,T);self.endpoint_algebraic_defect=float(abs(self.compatibility_operator).max());self.dependent_rate_defect=mul(T,self.A)-mul(self.parent_A,T);self.endpoint_boundary_kind='six_actual_native_rate_compatibility_equations'
