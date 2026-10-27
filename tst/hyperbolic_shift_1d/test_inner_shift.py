"""Native rate invariance, candidate speeds, and exact boundary-stage checks."""
import argparse,json
from pathlib import Path
import numpy as np
from evolution import Evolution
from puncture import PunctureEvolution
from principal import coefficient,symbol,speeds
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rng=np.random.default_rng(2727);errors=[]
for _ in range(100):
 r=rng.uniform(0,1);O=(1-r*r)/2;alpha,chi,g=np.exp(rng.uniform(-3,1,3));F=alpha**2+4*O*O*alpha;eta=1.0001*alpha/(alpha+1e-4*(1+r*r)/2);w=max(1-r/.8,0)
 mu=coefficient(alpha,chi,g,F,eta,w,.005,.02,inner_shift=.1,omega=O)
 P=symbol(alpha,chi,g,F,eta,w,.005,.02,mu);actual=np.linalg.eigvals(P);expected=speeds(alpha,chi,g,F,eta,w,.005,.02,mu)
 errors.append(float(max(abs(np.sort_complex(actual)-np.sort_complex(expected)))))
 assert errors[-1]<2e-7*max(1,np.max(abs(expected)))
 assert 4*mu/(3*g)>=2*(chi*F/g+eta*chi*.005*w*w/.02)*(1-1e-14)
r=np.logspace(-7,-4,20);chi=2*r**4;alpha=.7*r*r;O=(1-r*r)/2;F=alpha**2+4*O*O*alpha;eta=1.0001*alpha/(alpha+1e-4*(1+r*r)/2)
c=np.sqrt(4*coefficient(alpha,chi,1,F,eta,1,.005,.02,inner_shift=.1,omega=O)/3)
slope=float(np.polyfit(np.log(r),np.log(c),1)[0]);assert abs(slope-1)<1e-5
regular=Evolution(65,a.library,restoring=4,inner_shift=.1);q=np.zeros((65,10));q[:,0]=.001*np.exp(-((regular.r-.5)/.1)**2);q[:,1]=.002*np.exp(-((regular.r-.6)/.1)**2)
q[-1,0]=.01;q[-1,1]=-.01
for _ in range(3):q=regular.step(q,1e-5)
assert regular.stats['max_relative_null']<5e-14
puncture=PunctureEvolution(129,a.library,inner_shift=.1);q=puncture.initial();q[30:60,1]+=.0001;q[40,5]+=.0002
full=puncture.rhs(q,diagnostics=True);lean=puncture.rhs(q,diagnostics=False);assert np.array_equal(full,lean)
floored=PunctureEvolution(65,a.library,lapse_floor=2e-9);q=floored.initial();q[1:4,6]=-1
q=floored.materialize(q);assert np.array_equal(q[1:4,6],2e-9*floored.O[1:4]);assert q[-1,6]==1
result=dict(configurable_physical_lapse_floor=True,speed_max_absolute_error=max(errors),AF_shift_speed_log_slope=slope,perturbed_stage_null_max=regular.stats['max_relative_null'],diagnostic_omission_bitwise_equal_RHS=True,
 scope='Scalar frozen speeds and exact algebraic identities; not a uniform hyperbolicity or nonlinear convergence proof')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
