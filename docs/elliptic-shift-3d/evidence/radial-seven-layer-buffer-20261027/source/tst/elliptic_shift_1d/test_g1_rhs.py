#!/usr/bin/env python3
"""Independent agreement with the unreduced nonlinear tensor implementation."""
import argparse,json
from pathlib import Path
import numpy as np
from numerics import RadialOperators
from kernel import Kernel,PARITY
import g1_rhs
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path);a=p.parse_args();kernel=Kernel(a.library)
results=[]
for seed in range(6):
 r=(np.arange(80)+.5)/79.5;op=RadialOperators(r);rng=np.random.default_rng(seed);q=np.zeros((80,8))
 for field in [0,2,3,4,5,6]:
  coef=rng.normal(size=4)*.002
  q[:,field]=sum(coef[k]*r**(2*k) for k in range(4))
  if field==2:q[:,field]*=r*r
  if field==5:q[:,field]*=r
 q[-1,6]=0 # Fixed clock, while other off-constraint endpoint errors are retained.
 d,dd=op.derivatives(q,PARITY)
 for variant in [False,True]:
  for xi in [0.,1.]:
   R,S,c,b,db,ddb=g1_rhs.parts(op,q,d,dd,appendix_c=variant,restoring=xi)
   R0,S0,c0=kernel.parts(r,q,d,dd,b,db,ddb,appendix_c=variant,lapse_restoring=xi)
   O=(1-r*r)/2
   raw=R[:-1]+S[:-1]/O[:-1,None];native=R0[:-1]+S0[:-1]/O[:-1,None]
   error=float(np.max(abs(raw-native)));constraint_error=float(np.max(abs(c[:,:6]-c0[:,:6])))
   assert error<2e-12,(seed,variant,xi,error)
   assert constraint_error<5e-14,constraint_error
   assert np.max(abs(S[-1]-S0[-1]))<2e-14
   np.testing.assert_allclose(c[:-1,10],c0[:-1,10],atol=3e-10,rtol=3e-12)
   results.append(dict(seed=seed,appendix_c=variant,restoring=xi,max_rhs_error=error,max_constraint_error=constraint_error))
q=np.zeros((80,8));d,dd=op.derivatives(q,PARITY);R,S,c,*_=g1_rhs.parts(op,q,d,dd,restoring=1)
assert np.max(abs(R))==0 and np.max(abs(S))==0 and np.max(abs(c[:,:6]))==0
record={'kind':'full nonlinear spherical g=1 restriction with analytic cancellations; independent native-tensor agreement',
 'cases':results,'exact_equilibrium_regular_and_pole':True,'no_projection':'Theta,Lambda,trace and curvature remain free; g_t=0 is the exact elliptic metric gauge identity'}
if a.output:a.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'cases':len(results),'max_rhs_error':max(x['max_rhs_error'] for x in results),'exact_equilibrium':True}))
