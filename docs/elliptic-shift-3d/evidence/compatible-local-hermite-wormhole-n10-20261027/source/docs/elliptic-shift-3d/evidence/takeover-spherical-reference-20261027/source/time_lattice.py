"""Integer-step clock for autonomous fixed-CFL integration and exact reports."""
import math


def near_integer(value):
    nearest=round(value)
    return nearest if abs(value-nearest)<=8*math.ulp(value) else None


class FixedStepClock:
    def __init__(self,dt,t,step,target,metadata=None):
        if not dt>0 or target<t:raise ValueError('Invalid fixed-step clock')
        self.dt=float(dt);self.t=float(t);self.step=int(step);self.target=float(target)
        self.origin_time=float(metadata['origin_time']) if metadata else self.t
        self.origin_step=int(metadata['origin_step']) if metadata else self.step
        if metadata and metadata['dt']!=self.dt:raise ValueError('Checkpoint timestep lattice differs')
        represented=self.origin_time+(self.step-self.origin_step)*self.dt
        if abs(represented-self.t)>16*max(math.ulp(self.t),math.ulp(represented)):
            raise ValueError('Checkpoint time is not on its declared timestep lattice')
        quotient=(self.target-self.origin_time)/self.dt;nearest=near_integer(quotient)
        full=nearest if nearest is not None else math.floor(quotient)
        self.final_full_step=self.origin_step+full
        self.final_delta=0. if nearest is not None else self.target-(self.origin_time+full*self.dt)
        if self.final_full_step<self.step:raise ValueError('Target precedes the current timestep lattice')
        self.complete=self.step==self.final_full_step and self.final_delta==0
        if self.complete:self.t=self.target

    @property
    def full_steps_available(self):return self.final_full_step-self.step

    def report_step(self,time):
        quotient=(time-self.origin_time)/self.dt;nearest=near_integer(quotient)
        return self.origin_step+(nearest if nearest is not None else math.ceil(quotient))

    def advance(self,done,h):
        if done==0:return
        if self.complete:raise ValueError('Advancing a completed clock')
        if h==self.dt:
            if done>self.full_steps_available:raise ValueError('Stepping beyond the final full CFL step')
            self.step+=done;self.t=self.origin_time+(self.step-self.origin_step)*self.dt
            if self.step==self.final_full_step and self.final_delta==0:self.t=self.target;self.complete=True
        else:
            if self.full_steps_available or done!=1 or h!=self.final_delta:raise ValueError('Unexpected fractional landing')
            self.step+=1;self.t=self.target;self.origin_time=self.t;self.origin_step=self.step;self.complete=True

    def metadata(self):
        return dict(origin_time=self.origin_time,origin_step=self.origin_step,dt=self.dt,
                    convention='Integer full-step count; a fractional final landing starts a new lattice epoch.')
