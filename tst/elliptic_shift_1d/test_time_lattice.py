#!/usr/bin/env python3
"""Long integer clocks, exact report ticks, aligned and fractional restarts."""
import math,json
from time_lattice import FixedStepClock
records=[]
for n in [64,512,1024,2048]:
 for cfl in [.1,.05,.025,.0125]:
    dt=cfl/(n-.5);clock=FixedStepClock(dt,0.,0,6.)
    for tick in range(1,61):
        target=tick*.1;step=clock.report_step(target);count=step-clock.step
        clock.advance(count,dt)
        assert -16*math.ulp(target)<=clock.t-target<dt+16*math.ulp(target),(n,cfl,tick,clock.t,target)
        if cfl<=.05 or tick%2==0:assert abs(clock.t-target)<16*math.ulp(target)
    assert clock.complete and clock.t==6.
    parent=FixedStepClock(dt,0.,0,1.);parent.advance(parent.full_steps_available,dt)
    child=FixedStepClock(dt,parent.t,parent.step,6.,parent.metadata());child.advance(child.full_steps_available,dt)
    assert (child.t,child.step,child.metadata())==(clock.t,clock.step,clock.metadata())
    records.append(dict(n=n,CFL=cfl,aligned_reports_and_restart=True))
# About one hundred million steps are represented by integer counts, without
# a hundred million floating additions. Report times are exact epoch products.
dt=.1/1023.5;clock=FixedStepClock(dt,0.,0,10000.)
while clock.full_steps_available:clock.advance(min(4096,clock.full_steps_available),dt)
assert clock.complete and clock.t==10000. and clock.step==102350000
# A genuine fractional final step starts a new lattice epoch; it is not
# falsely asserted to equal an unsplit trajectory with different RK steps.
clock=FixedStepClock(.003,0.,0,.010);clock.advance(3,.003);assert not clock.complete
clock.advance(1,clock.final_delta);assert clock.complete and clock.t==.010
child=FixedStepClock(.003,clock.t,clock.step,.040,clock.metadata());child.advance(child.full_steps_available,.003)
assert child.complete and child.t==.040 and child.step==14
# Invalid checkpoint/timestep metadata must be rejected.
try:FixedStepClock(.004,.010,4,.040,clock.metadata())
except ValueError:pass
else:raise AssertionError('A changed CFL was accepted')
print(json.dumps(dict(cases=records,long_steps=102350000,fractional_epoch=True,changed_timestep_rejected=True)))
