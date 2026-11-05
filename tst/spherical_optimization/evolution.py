"""Spatial/RK policies around the C++ current-formulation continuum kernels.

Seven-point polynomial radial derivatives and sign-selected biased advection.
Puncture runs use one-sided stencils, never a fictitious regular-center parity.
No stationary-background subtraction and no elliptic solve.
"""
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from numerics import RadialOperators,weights,puncture_dissipation
from kernel import Kernel,PARITY

class Evolution:
    def __init__(self,kernel,n,grading=0.,penalty=.6510416666660156,ko=.5,floors=False):
        self.kernel=kernel;self.n=n;self.penalty=penalty;self.ko=ko;self.floors=floors
        x=(np.arange(n)+.5)/(n-.5)
        # Monotone smooth grading: finer toward scri. Actual dr enters all CFLs.
        if not 0<=grading<1:raise ValueError('grading must lie in [0,1)')
        self.r=x+grading*x*(1-x);self.r[-1]=1
        self.dr=np.r_[np.diff(self.r),self.r[-1]-self.r[-2]]
        self.h=float(min(self.dr));self.op=RadialOperators(self.r,width=7,puncture=True,subtract_local=True)
        self.initial,self.source,self.omega=kernel.initial(self.r)
        self.normal=weights(self.r[-7:],1.,1)
        self.calls=0;self.floor_calls=0;self.floor_max_r=0.;self.last={};self.max_speed=0.
        self.cumulative_rhs_seconds=0.;self.step_speed_over_dr=0.;self.worst_speed_over_dr=0.
    def materialize(self,state):
        q=state.copy()
        if not np.isfinite(q).all():raise FloatingPointError('nonfinite state')
        if self.floors:
            chi=(q[:,0]<1e-8)&(self.r<.03);alpha=q[:,8]<1e-10*self.omega[:,0]
            active=chi|alpha
            if np.any(active):self.floor_calls+=1;self.floor_max_r=max(self.floor_max_r,float(max(self.r[active])))
            q[chi,0]=1e-8;q[alpha,8]=1e-10*self.omega[alpha,0]
        if np.any(q[:,[0,1,2,8]]<=0):
            i,f=np.argwhere(q[:,[0,1,2,8]]<=0)[0];raise FloatingPointError(f'nonpositive field {f} at r={self.r[i]}')
        q[-1,8]=1.;q[-1,9]=-np.sqrt(q[-1,0]/q[-1,1]);return q
    def rhs(self,state,record=False):
        started=time.perf_counter();self.attempted_state=state.copy();q=self.materialize(state)
        d,dd=self.op.derivatives(q,PARITY)
        R,S,traces,diag=self.kernel.parts(self.r,q,d,dd,self.source)
        rate=np.zeros_like(q);rate[:-1]=R[:-1]+S[:-1]/self.omega[:-1,:1]
        normal=self.normal@(S[-7:]-S[-1])
        rate[-1],dependent=self.kernel.scri(q[-1],d[-1],dd[-1],normal,self.dr[-1],self.penalty)
        q[-1,11]=dependent[0]
        adv=self.op.advection_derivative(q,PARITY,q[:,9])
        # Common change from centered to biased component-advection derivative.
        rate[:-1,:11]+=q[:-1,9,None]*(adv[:-1,:11]-d[:-1,:11])
        # KO is a disclosed numerical policy; taper unchanged by candidate.
        diss=puncture_dissipation(q,self.r,self.h,self.ko,.8,order=8,include_first=True)
        rate[:-1,:11]+=diss[:-1,:11]
        # Feed the ACTUAL Lambda RHS and selected advective jets to shared D rate.
        auxiliary=np.zeros(self.n)
        self.kernel.check(self.kernel.lib.spherical_auxiliary(self.kernel.handle,self.n,self.r,q,
            np.ascontiguousarray(adv),np.ascontiguousarray(rate[:,7]),auxiliary))
        rate[:-1,11]=auxiliary[:-1]+diss[:-1,11]
        self.calls+=1;self.max_speed=max(self.max_speed,float(max(diag[:,7])))
        self.cumulative_rhs_seconds+=time.perf_counter()-started
        if not np.isfinite(rate).all():raise FloatingPointError('nonfinite RHS')
        self.last=dict(q=q,d=d,dd=dd,diag=diag,pole=S[-1],trace=traces[-1],
                       clock=dependent[1],dependent_D=dependent[0],auxiliary_trace_pde=auxiliary[-1],
                       speed_over_dr=float(max(diag[:,7]/self.dr)),speed=float(max(diag[:,7])))
        self.step_speed_over_dr=max(self.step_speed_over_dr,self.last['speed_over_dr'])
        self.worst_speed_over_dr=max(self.worst_speed_over_dr,self.last['speed_over_dr'])
        return rate
    def step(self,q,dt):
        self.step_speed_over_dr=0.
        a=self.rhs(q);b=self.rhs(q+dt*a/2);c=self.rhs(q+dt*b/2);d=self.rhs(q+dt*c)
        result=q+dt*(a+2*b+2*c+d)/6
        self.rhs(result);return self.last['q'].copy()
    def report(self,q,t):
        self.rhs(q);x=self.last;diag=x['diag'];w=self.r**2*self.dr
        out=dict(t_M=3*t,calls=self.calls,speed=x['speed'],speed_over_dr=x['speed_over_dr'],
          determinant_max=float(max(abs(diag[:,4]))),tracefree_max=float(max(abs(diag[:,5]))),clock=float(abs(x['clock'])),pole=float(max(abs(x['pole']))),trace=float(max(abs(x['trace']))),
          null=float(abs(x['q'][-1,0]/x['q'][-1,1]-x['q'][-1,9]**2)),
          dependent_D=float(x['dependent_D']),auxiliary_trace_pde=float(x['auxiliary_trace_pde']),
          floors=self.floor_calls,floor_max_r=self.floor_max_r,
          alpha_min=float(min(q[:,8])),chi_min=float(min(q[:,0])),rhs_seconds=self.cumulative_rhs_seconds)
        for lo,hi,label in [(.65,.995,'exterior'),(.8,.9,'join'),(.9,.97,'tail'),(.97,.995,'scri_collar')]:
            mask=(self.r>=lo)&(self.r<=hi)
            for f,name in enumerate(('H','M','Theta','Z')):
                out[label+'_'+name]=float(np.sqrt(np.sum(w[mask]*diag[mask,f]**2)/sum(w[mask]))) if np.any(mask) else None
        expansion=diag[:,9];roots=np.flatnonzero((expansion[:-1]<0)&(expansion[1:]>=0))
        out['horizon_accepted']=bool(len(roots));out['horizon_mass']=None;out['horizon_r']=None
        if len(roots):
            j=roots[-1];fraction=-expansion[j]/(expansion[j+1]-expansion[j])
            out['horizon_r']=float(self.r[j]+fraction*(self.r[j+1]-self.r[j]))
            out['horizon_mass']=float(diag[j,8]+fraction*(diag[j+1,8]-diag[j,8]))
        out['lapse_gradient_max']=float(max(abs(x['d'][:,8])))
        out['shift_gradient_max']=float(max(abs(x['d'][:,9])))
        return out
