"""Independent finite-relaxation identities and coupled-RK order checks."""
from pathlib import Path
import re
import numpy as np
import sympy as s
from verify_smooth_pulses import rate_gradient, integrated_rate, prediction
from verify_pulses import prediction as constant_prediction

x,y,z,tau=s.symbols('x y z tau', real=True)
xyz=[x,y,z]
E=s.Matrix([x*x+y*z, x*z+y*y, x*y+z*z])
lam=2+x*x+y*y+z*z
for i,j in [(0,1),(0,2),(1,2)]:
    curl=s.diff(E[j],xyz[i])-s.diff(E[i],xyz[j])
    source=s.diff(-lam*E[j],xyz[i])-s.diff(-lam*E[i],xyz[j])
    assert s.expand(source+lam*curl+s.diff(lam,xyz[i])*E[j]-s.diff(lam,xyz[j])*E[i])==0
    mapped=s.exp(-tau*lam)*E
    mapped_curl=s.diff(mapped[j],xyz[i])-s.diff(mapped[i],xyz[j])
    exact=s.exp(-tau*lam)*(curl-tau*(s.diff(lam,xyz[i])*E[j]-s.diff(lam,xyz[j])*E[i]))
    assert s.simplify(mapped_curl-exact)==0
print('PASS: finite relaxation curl map and negative d(lambda) wedge E source, all three pairs')

pc=dict(reduction_rate='2',reduction_profile='smooth_core',reduction_inner_rate='16',
        reduction_core_radius='.2',reduction_taper_radius='.8')
points=np.random.default_rng(943).uniform(-.7,.7,(256,3))
rate,gradient=rate_gradient(points,pc,0,.5)
assert np.all((rate>=2)&(rate<=16))
epsilon=1e-6
for d in range(3):
    step=np.eye(3)[d]*epsilon
    numerical=(rate_gradient(points+step,pc,0,.5)[0]-rate_gradient(points-step,pc,0,.5)[0])/(2*epsilon)
    assert np.max(abs(gradient[:,d]-numerical))<2e-7
print('PASS: independent smooth profile gradient, bounded rates, deterministic random samples')
integral, grad_integral = integrated_rate(points,.5,pc,.5,256)
endpoint = points+np.array([.25,0,0])
fundamental = (rate_gradient(endpoint,pc,0,.5)[0]-rate)/.5
assert max(abs(grad_integral[:,0]-fundamental))<1e-9
moving = dict(pc,reduction_follow_trackers='true',co_0='true',co_1='true',
              co_0_x='-.25',co_1_x='.25')
mr,mg = rate_gradient(points,moving,.5,.5)
ir,ig = rate_gradient(endpoint,moving,0,.5)
swapped = dict(moving,co_0_x='.25',co_1_x='-.25')
sr,sg = rate_gradient(points,swapped,.5,.5)
assert np.max(abs(mr-ir))<1e-13 and np.max(abs(mg-ig))<1e-12
assert np.max(abs(mr-sr))<1e-13 and np.max(abs(mg-sg))<1e-12
assert np.all((mr>=2)&(mr<=16))
print('PASS: integrated gradient versus endpoint identity; bounded moving union and center permutation')

# All 33 component encodings must reduce to the previously checked constant-rate solution.
data=np.zeros(129,dtype=[(name,float) for name in ['time','x','y','z']])
data['x']=np.linspace(-1,1,129);data['time']=.3
for family in ['p','Q','L','B']:
    for direction in [0,1,2]:
        problem=dict(pulse_family=family,pulse_direction=str(direction))
        params=dict(problem=problem,pc_gh=dict(reduction_rate='2'))
        new,curl=prediction(data,params)
        old=constant_prediction(data,problem,2)
        assert np.max(abs(new-old))<1e-22
print('PASS: characteristic prediction has identical constant-rate limit for every pulse family/direction')

# Coupled nonlinear ODE: u'=u, X'=-u X has X=exp(1-exp(t)). Read the actual
# driver coefficients, but compare against this independently integrated flow.
root=Path(__file__).resolve().parents[2]
driver=(root/'src/driver/driver.cpp').read_text()
chunk=driver.split('} else if (integrator == "rk4")')[1].split('} else if')[0]
coeff={name:np.array([float(re.search(rf'{name}\[{n}\] = ([^;]+);',chunk)[1])
                     for n in range(4)]) for name in ['gam0','gam1','beta','delta']}
errors=[]
for steps in [10,20,40,80]:
    state=np.array([1.,1.]);dt=.5/steps
    for _ in range(steps):
        accumulator=state.copy()
        for stage in range(4):
            if stage: accumulator+=coeff['delta'][stage]*state
            rhs=np.array([state[0],-state[0]*state[1]])
            state=coeff['gam0'][stage]*state+coeff['gam1'][stage]*accumulator+dt*coeff['beta'][stage]*rhs
    errors.append(float(abs(state[1]-np.exp(1-np.exp(.5)))))
orders=np.log2(np.array(errors[:-1])/errors[1:])
assert min(orders)>3.8, (errors,orders)
print('PASS: coupled nonlinear RK4 flow; errors=',errors,'orders=',orders.tolist())
