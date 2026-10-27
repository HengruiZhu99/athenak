"""Independent tensor checks of the new total adapter and AF leading rates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from puncture import PunctureEvolution
from kernel import Kernel
from gauge import collapsed_background_correction,restoring_coefficient
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
k=Kernel(a.library);f=k.lib.spherical_parts_positive_totals;f.argtypes=k.lib.spherical_parts.argtypes
r=np.array([1e-3,5e-4,1e-4,1e-5]);q=np.zeros((4,8));d=q.copy();dd=q.copy();A=(1+r*r)/2;O=(1-r*r)/2
q[:,0]=2*r**4;d[:,0]=8*r**3;dd[:,0]=24*r*r
q[:,1]=.2*r+.1*r*r;d[:,1]=.2+.2*r;dd[:,1]=.2
for j,c in [(2,.3),(3,.4),(4,.1)]:q[:,j]=c*r*r;d[:,j]=2*c*r;dd[:,j]=2*c
q[:,5]=.05+.01*r;d[:,5]=.01;q[:,6]=.7*r*r;d[:,6]=.4*r;dd[:,6]=.4
beta=-.9*r+.03*r*r;db=-.9+.06*r;ddb=np.full(4,.06);R=np.zeros((4,8));S=R.copy();diag=np.zeros((4,12))
assert f(4,r,q,d,dd,beta,db,ddb,4.,-.99,1,1.,R,S,diag)==0
S[:,6]-=2*restoring_coefficient(r,1,profile='physical-log')*q[:,6]*(q[:,6]-A)
S[:,6]+=collapsed_background_correction(r,q[:,6]-A);x=R+S/O[:,None]
observed=np.c_[x[:,0]/q[:,0],x[:,6]/q[:,6],x[:,5],x[:,1]/r,x[:,2:5]/r[:,None]**2]
expected=np.array([-1.8,-1.8,.79,-.14,-.54,-.72,-.18])
errors=np.max(abs(observed-expected),axis=1)
assert errors[-1]<3e-6 and np.all(np.diff(errors)<0)
e=PunctureEvolution(129,a.library);state=e.initial();rate,projected=e.rhs(state,return_state=True)
# Evolved connection stays independent even when deliberately perturbed.
perturbed=state.copy();perturbed[40,5]+=.001
_,retained=e.rhs(perturbed,return_state=True);assert retained[40,5]==perturbed[40,5]
assert np.any(abs(rate[1:-1,1])>1e-12),'metric must evolve'
record=dict(AF_limit_radii=r.tolist(),AF_errors=errors.tolist(),AF_observed=observed.tolist(),
 AF_expected=expected.tolist(),initial_report=e.report(state,0.),
 independent_connection_retained=True,live_metric=True,
 scope='Leading AF asymptotics and initialization; not convergence of puncture evolution')
a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(AF_errors=errors.tolist(),live_metric=True)))
