"""Independent AF forcing, complete scri limit, and native-rate controls."""
import argparse,json
from pathlib import Path
import numpy as np
from evolution import Evolution
from puncture import PunctureEvolution
from shift_source import background
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=np.logspace(-3,-6,4);A=(1+r*r)/2;O=(1-r*r)/2;alpha=.7*r*r;u=alpha-A;du=.4*r
b1=.3;b2=.02;beta=b1*r+b2*r*r;db=b1+2*b2*r;xi=4
source,delta,dp=background(r,alpha,u,du,'lapse-adapted')
rate=beta*db+source-xi*(beta+r+delta)/O
errors=abs(rate/r-(b1*b1-2*xi*b1));assert np.all(np.diff(errors)<0) and errors[-1]<1e-6
r=1-np.logspace(-3,-6,4);A=(1+r*r)/2;O=(1-r*r)/2;u=.17*(r-1)+.09*(r-1)**2;du=.17+.18*(r-1);alpha=A+u
_,delta,delta_prime=background(r,alpha,u,du,'lapse-adapted');c=.23;d=.12
beta=-r*alpha/A+c*(r-1)+d*(r-1)**2
limit_error=abs(-xi*(beta+r+delta)/O-xi*c);assert limit_error[-1]<2e-6 and np.all(np.diff(limit_error)<0)
e=Evolution(65,a.library,restoring=4,shift_source='lapse-adapted');z=np.zeros((65,10));rate=e.rhs(z);equilibrium=float(np.max(abs(rate)));assert equilibrium<2e-9
checks=[]
for amp in [-.01,.001,.01]:
 e=Evolution(33,a.library,restoring=4,shift_source='lapse-adapted');q=np.zeros((33,10));q[:,0]=amp*e.r**2;q[:,1]=.4*amp*e.r**2;q[:,6]=.3*amp*e.r**2
 q=e.step(q,1e-8);assert e.stats['max_relative_null']<2e-15 and e.stats['max_driver_boundary_defect']<1e-12
 checks.append(dict(amplitude=amp,**e.stats))
a0=PunctureEvolution(129,a.library);a1=PunctureEvolution(129,a.library,shift_source='lapse-adapted');q=a0.initial();q[35:55,1]+=.0001;q[47,5]+=.0002
r0=a0.rhs(q);r1=a1.rhs(q);assert np.array_equal(r0[:,:7],r1[:,:7]),'Lapse and Einstein rates must remain unchanged for the same state'
from telegraph import window
rr=np.linspace(0,1,129);AA=(1+rr*rr)/2;ww,wp=window(rr,width=.24,inner_profile='af-regular')
ss,delta,dp=background(rr,.3*AA,-.7*AA,-.7*rr,'lapse-inner',ww,wp)
outer=rr>=.8
assert np.array_equal(ss[outer],-rr[outer]) and np.all(delta[outer]==0) and np.all(dp[outer]==0)
ei=Evolution(33,a.library,restoring=4,shift_source='lapse-inner');z=np.zeros((33,10));assert np.max(abs(ei.rhs(z)))<2e-9
z[:,0]=.001*ei.r**2;z[:,1]=-.001*ei.r**2;z[:,6]=.002*ei.r**2;ei.step(z,1e-8)
assert ei.stats['max_relative_null']<2e-15 and ei.stats['max_driver_boundary_defect']<1e-12
result=dict(inner_collar_exactly_unchanged=True,AF_slope_expected=b1*b1-2*xi*b1,AF_errors=errors.tolist(),scri_limit_errors=limit_error.tolist(),Minkowski_max_RHS=equilibrium,perturbed_stage_checks=checks,unchanged_native_and_lapse_rates=True,
 scope='Lower-order source change: interior principal matrix unchanged. Boundary auxiliary limit rederived; no claim of nonlinear duration stability.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='perturbed_stage_checks'}))
