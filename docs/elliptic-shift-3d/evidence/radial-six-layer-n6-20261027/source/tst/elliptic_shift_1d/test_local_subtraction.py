#!/usr/bin/env python3
"""Constant-nullspace arithmetic and independent band-matrix equivalence."""
import json
import numpy as np
from numerics import RadialOperators
from shift_ode import BandedShift
records=[]
for grid in ['half','nodal','cutoff']:
 for n in [33,65]:
  r=(np.arange(n)+.5)/(n-.5) if grid=='half' else np.linspace(0 if grid=='nodal' else .03,1,n)
  op=RadialOperators(r,puncture=grid=='cutoff',subtract_local=True)
  # Constants are annihilated exactly, including one-sided closures/transport.
  values=np.full((n,3),.013)
  d,dd=op.derivatives(values,np.ones(3))
  assert not np.any(d) and not np.any(dd)
  assert op.endpoint_first(values[:,0])==0
  for beta in [-np.ones(n),np.ones(n)]:assert not np.any(op.advection_derivative(values,np.ones(3),beta))
  for degree in [0,2,4,6]:
   values=r**degree;d,dd=op.derivatives(values,1)
   expected=np.zeros(n) if degree==0 else degree*r**(degree-1)
   expected2=np.zeros(n) if degree==0 else degree*(degree-1)*r**(degree-2)
   np.testing.assert_allclose(d,expected,atol=2e-11,rtol=0)
   np.testing.assert_allclose(dd,expected2,atol=1e-8,rtol=0)
  for bias in ['centered','right']:
   solver=BandedShift(op,bias=bias);b=solver.bandwidth
   # Public derivative on basis vectors independently includes local subtraction.
   dense=op.derivatives(np.eye(n),np.ones(n))[0] if bias=='centered' else op.advection_derivative(np.eye(n),np.ones(n),np.ones(n))
   dense=np.vstack([dense[solver.skip:n-1+solver.skip]*solver.scale,np.eye(n)[-1]])
   actual=np.zeros((n,n))
   for i in range(n):
    for j in range(max(0,i-b),min(n,i+b+1)):actual[i,j]=solver.matrix[i,b+j-i]
   np.testing.assert_allclose(actual,dense,atol=2e-15,rtol=2e-15)
   W=.01*np.cos(3*r*r);source=-.06*r*np.sin(3*r*r);got=solver.solve(source,W[-1])
   np.testing.assert_allclose(got,np.linalg.solve(dense,np.r_[source[solver.skip:n-1+solver.skip]*solver.scale,W[-1]]),atol=2e-13,rtol=0)
   records.append(dict(grid=grid,n=n,bias=bias,continuum_error=float(max(abs(got-W)))))
print(json.dumps(dict(exact_constant_nullspace=True,independent_dense_matrix_equivalence=True,records=records)))
