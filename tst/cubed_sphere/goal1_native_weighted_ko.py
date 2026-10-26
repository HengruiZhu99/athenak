"""Separate native8 weighted radial numerical KO candidate, no added fields.

The CK01 parent and its checkpoints remain immutable. This is currently an
operator/Jacobian candidate, not an admitted general scri or nonlinear method.
"""
import numpy as np
from goal1_native_weighted_linear import Model as Parent,mul,native
class Model(Parent):
 def __init__(self,N,H=.08,weak=True,eta=8,diss=.1):
  super().__init__(N,H,weak,eta);D=np.zeros((self.M-3,self.M));weights=[]
  for i in range(self.M-3):
   nodes=self.o[i:i+4];D[i,i:i+4]=native.weights(nodes,nodes[0],3,self.h)*self.h**3;weights.append(float(np.mean(nodes)**2/H**2))
  edges=np.r_[0,(self.o[:-1]+self.o[1:])/2,H];Mrel=np.diff(edges)/self.h;Q=-diss/(64*self.h)*np.einsum('ki,k,kj->ij',D,np.array(weights),D)/Mrel[:,None];full=np.kron(Q,np.eye(8));self.KO_operator=mul(full[self.selected],self.expand_matrix);self.A+=self.KO_operator;self.boundary_source+=mul(full[self.selected],self.boundary_full);self.KO_scalar=Q;self.KO_mass=Mrel
