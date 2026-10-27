#!/usr/bin/env python3
"""Differentiate unreduced native pole numerators independently at scri."""
import argparse,json
import numpy as np
from kernel import Kernel
from numerics import RadialOperators
import g1_rhs,scri_limits
p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args();kernel=Kernel(a.library)
records=[]
for appendix in [False,True]:
 errors=[]
 for n in [16,32,64]:
  r=(np.arange(n)+.5)/(n-.5);op=RadialOperators(r,width=9,quadrature_width=11)
  q=np.zeros((n,8));d=q.copy();dd=q.copy();rng=np.random.default_rng(520)
  for f,base in [(0,0),(2,2),(3,0),(4,0),(5,1),(6,0)]:
   for k in range(3):
    power=2*k+base;coef=rng.normal()*.003
    q[:,f]+=coef*r**power
    if power:d[:,f]+=coef*power*r**(power-1)
    if power>=2:dd[:,f]+=coef*power*(power-1)*r**(power-2)
  timing={};R,S,diag,b,db,ddb=g1_rhs.parts(op,q,d,dd,appendix_c=appendix,restoring=.4,timing=timing)
  native_R,native,_=kernel.parts(r,q,d,dd,b,db,ddb,appendix_c=appendix,lapse_restoring=.4)
  # Clock-incompatible/off-constraint traces deliberately remain nonzero.
  fd=op.d1[-1]@native[-op.width:]
  exact,_=scri_limits.derivatives(q[-1],d[-1],dd[-1],timing['beta_deviation'][-1],appendix_c=appendix,restoring=.4)
  # R/S decompositions differ by analytic Omega multiples after cancellation.
  # The complete l'Hopital limit, rather than S' alone, must agree.
  errors.append(float(np.max(abs((native_R[-1]-fd)-(R[-1]-exact)))))
 orders=np.log2(np.array(errors[:-1])/np.array(errors[1:]))
 assert errors[-1]<2e-11 and min(orders)>5,(errors,orders)
 records.append(dict(appendix_c=appendix,errors=errors,orders=orders.tolist()))
# Verify the directional damping action independently, retaining all traces.
h=1/127.5;gamma=4.;lift=1/op.d1[-1,-2]
correction,G=scri_limits.corner_correction(q[-1],d[-1],dd[-1],R[-1],timing['beta_deviation'][-1],h,gamma,lift,restoring=.4)
epsilon=1e-6;perturbed=d[-1].copy();perturbed[[0,3]]+=epsilon*correction/lift
Rperturbed=R[-1].copy();Rperturbed[0]+=b[-1]*epsilon*correction[0]/lift
ds,_=scri_limits.derivatives(q[-1],perturbed,dd[-1],timing['beta_deviation'][-1],restoring=.4)
Gp=(Rperturbed-ds)[[0,6]]
np.testing.assert_allclose((Gp-G)/epsilon,-gamma/h*G,rtol=2e-8,atol=2e-9)
print(json.dumps(records))
