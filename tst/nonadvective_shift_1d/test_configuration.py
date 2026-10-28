"""Independent algebraic and implementation oracles; no trajectory launch."""
import argparse,json
from pathlib import Path
import numpy as np
from candidate import NonadvectiveEvolution
from nonadvective_gauge import coefficient,separation,shift_speeds,incoming_vector,symbol
from principal import eigenspace_audit
from boundary import normal_shift,normal_rate
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
checks=[];rng=np.random.default_rng(19384)
for mode in [0.,1.]:
 e=NonadvectiveEvolution(65,a.library,puncture=False,reference_damping=mode,reference_rate=.1 if mode else 0)
 q=e.initial();rhs=e.rhs(q);maximum=float(np.max(abs(rhs)));assert maximum<2e-9
 checks.append(dict(name='Minkowski discrete equilibrium',eta=mode,max_RHS=maximum))
 # Test boundary preservation at finite metric perturbations and the incoming
 # eigenvector identity, including its connection-constraint-null direction.
 for amp in [-.002,.001,.003]:
  q=e.initial();q[:,0]+=amp*e.r**2;q[:,1]+=.4*amp*e.r**2;q[-1,9]+=.2*amp
  before=e.base.materialize(q);x=e.geometry(before);after,final=e.project(before)
  lam,v=incoming_vector(x['chi'][-1],x['g'][-1],x['beta'][-1],x['mu'][-1])
  actual=after[-1,[10,11,5,12]]-before[-1,[10,11,5,12]]
  assert np.allclose(actual,actual[2]*v[[0,1,5,7]],atol=2e-15,rtol=2e-13)
  L=symbol(x['alpha'][-1],x['chi'][-1],x['g'][-1],x['alpha'][-1]**2,1.,0.,.005,.02,x['mu'][-1],x['beta'][-1])
  assert np.max(abs(L@v-lam*v))<3e-14
  assert abs(actual[2]-actual[1]/x['g'][-1]**2)<2e-15
  assert np.max(abs(e.materialize(after)-after))<5e-13
  rate=e.rhs(after);z=e.last;expected=normal_rate(z['alpha'][-1],z['chi'][-1],z['g'][-1],0.,rate[-1,0],rate[-1,1])
  assert abs(rate[-1,7]-expected)<2e-13
  assert e.stats['max_relative_null']<2e-15 and abs(final['defect'])<2e-12
 checks.append(dict(name='perturbed metric boundary projection',eta=mode,incoming_only=True,connection_constraint_trace_preserved=True,idempotent=True))
 # Bulk equations must have neither beta advection nor old -r/1-Omega terms.
 e=NonadvectiveEvolution(129,a.library,puncture=True,reference_damping=mode,reference_rate=.1 if mode else 0,ko=0)
 q=e.initial();q[:,5]+=.001*e.r;q[:,9]-=.003*e.r
 f=e.rhs(q);s=e.materialize(q);bulk=(e.r>0)&(e.r<=.8)
 assert np.array_equal(f[bulk,7],4*s[bulk,5]-mode*(s[bulk,7]-s[bulk,9]))
 assert np.array_equal(f[bulk,9],e.reference_rate*(s[bulk,7]-s[bulk,9]))
 assert s[47,5]==q[47,5]
 # Compare all unchanged physical/telegraph RHS entries away from boundary.
 initial=e.initial();old=e.base.rhs(initial[:,:10]);new=e.rhs(initial)
 inner=(e.r>0)&(e.r<.7)
 assert np.array_equal(old[inner,:7],new[inner,:7])
 assert np.array_equal(old[inner,8],new[inner,8])
 checks.append(dict(name='bulk equations and unchanged Einstein/lapse/telegraph rates',eta=mode,exact=True))
for i in range(100):
 alpha,chi,g=np.exp(rng.uniform(-2,1,3));beta=rng.uniform(-2,2);F=alpha*alpha+rng.uniform(.1,2)*alpha;eta=rng.uniform(.1,1);w=rng.uniform(0,1)
 cl2=chi*F/g+eta*chi*.25*w*w;mu=.75*g*(2*cl2+beta*beta)
 L=symbol(alpha,chi,g,F,eta,w,.005,.02,mu,beta);light=alpha*np.sqrt(chi/g);cl=np.sqrt(cl2);vm,vp=shift_speeds(beta,g,mu)
 expected=np.array([beta,beta-light,beta-light,beta+light,beta+light,beta-cl,beta+cl,-vm,-vp])
 audit=eigenspace_audit(L,expected,tolerance=1e-7);assert audit['complete'],audit
 assert separation(beta,g,mu,cl2)>0
checks.append(dict(name='100 randomized scalar symbols',complete=True))
result=dict(checks=checks,passed=len(checks),scope='Algebraic/RHS preparation only; short and long trajectories remain separately gated.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
