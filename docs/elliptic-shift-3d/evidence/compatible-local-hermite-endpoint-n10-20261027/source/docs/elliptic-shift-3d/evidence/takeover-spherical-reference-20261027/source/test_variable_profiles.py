#!/usr/bin/env python3
"""Independent finite-difference jets and exactly flat outer profile."""
import json
import numpy as np
from variable_profiles import trumpet_metric_profile
r=np.linspace(.001,1,257);compact=trumpet_metric_profile(r,True);outer=r>=.3
assert np.array_equal(compact[outer,:6],np.tile([1,0,0,0,0,0],(sum(outer),1)))
records=[]
for variant in [False,True]:
 for x in [.05,.15,.2,.25,.4,.95]:
  h=1e-5;nodes=x+h*np.arange(-2,3);data=trumpet_metric_profile(nodes,variant);center=data[2]
  for column,expected,expected2 in [(0,center[1],center[2]),(3,center[4],center[5])]:
   f=data[:,column];first=(f[0]-8*f[1]+8*f[3]-f[4])/(12*h);second=(-f[0]+16*f[1]-30*f[2]+16*f[3]-f[4])/(12*h*h)
   assert abs(first-expected)<1e-7 and abs(second-expected2)<2e-4,(variant,x,column,first-expected,second-expected2)
  f=np.log(data[:,0]/nodes**2);ell=(f[0]-8*f[1]+8*f[3]-f[4])/(12*h);ellp=(-f[0]+16*f[1]-30*f[2]+16*f[3]-f[4])/(12*h*h)
  assert abs(ell-center[6])<1e-6 and abs(ellp-center[7])<2e-4,(variant,x,ell-center[6],ellp-center[7])
  records.append(dict(compact=variant,r=x))
np.testing.assert_allclose(compact[:,0]-1,compact[:,3],atol=4e-16,rtol=0)
print(json.dumps(dict(analytic_profile_jets_independently_verified=True,exact_flat_outer_frame=True,records=records,scope='Coordinate-only positive profiles, not black-hole sources or a constraint solve')))
