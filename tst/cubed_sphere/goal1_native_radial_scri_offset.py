"""CK02 arithmetic candidate: native offsets from the immutable reference.

Eight native spherical components, without additional evolved fields. The RHS is
unchanged and unsubtracted: d(u-u_ref)/dt=F(u), since u_ref is time independent.
The conditional smooth endpoint DAE and its known weak-class failure are retained.
"""
import numpy as np
from goal1_native_radial_scri import API,api,ACTIVE,CONFIG22,ROWS22,Model as TotalModel
class Model(TotalModel):
 def expand(self,z):
  return super().expand(self.ref[:,0,ACTIVE]+z)
 def enforce(self,z):
  full=self._enforce_total(z)
  answer=full-self.ref[:,0,ACTIVE]
  # The smooth DAE constrains six endpoint values. It never projects interior
  # states or the two live endpoint metric components onto reference values.
  answer[:-1]=z[:-1];answer[-1,[0,6]]=z[-1,[0,6]]
  return answer
 def _enforce_total(self,z):
  # Use a separate stateless view of the same grid/operators to evaluate the
  # old DAE in total variables; no extra Cauchy state is integrated.
  total=object.__new__(TotalModel);total.__dict__=self.__dict__
  return TotalModel.enforce(total,self.ref[:,0,ACTIVE]+z)
 def initial(self,epsilon):
  return self.enforce(api().oracle(self.r,0.,epsilon)[0][:,0,ACTIVE]-self.ref[:,0,ACTIVE])
