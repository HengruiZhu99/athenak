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
from physical_constraints import evaluate as physical_constraints

class Evolution:
    def __init__(self,kernel,n,grading=0.,penalty=.6510416666660156,ko=.5,floors=False,normal_policy="sampled",reference_jets=False):
        self.kernel=kernel;self.n=n;self.penalty=penalty;self.ko=ko;self.floors=floors;self.normal_policy=normal_policy
        x=(np.arange(n)+.5)/(n-.5)
        # Monotone smooth grading: finer toward scri. Actual dr enters all CFLs.
        if not 0<=grading<1:raise ValueError('grading must lie in [0,1)')
        self.r=x+grading*x*(1-x);self.r[-1]=1
        self.dr=np.r_[np.diff(self.r),self.r[-1]-self.r[-2]]
        self.h=float(min(self.dr));self.op=RadialOperators(self.r,width=7,puncture=True,subtract_local=True)
        self.initial,self.source,self.omega=kernel.initial(self.r)
        self.reference_jets=reference_jets;self.reference_jet_difference=None
        if reference_jets:
            derivatives=[]
            for step in (1e-3,5e-4):
                nodes=[];first=[];second=[]
                for r in self.r:
                    offsets=np.arange(-3,4) if r<1 else np.arange(-6,1)
                    h=min(step,r/8,(1-r)/8) if r<1 else step
                    x=r+h*offsets;nodes.extend(x)
                    first.append(weights(x,r,1));second.append(weights(x,r,2))
                samples,_,_=kernel.initial(np.ascontiguousarray(nodes));samples=samples.reshape(self.n,7,12)-self.initial[:,None,:]
                derivatives.append((np.einsum('ij,ijf->if',np.array(first),samples),np.einsum('ij,ijf->if',np.array(second),samples)))
            self.reference_d,self.reference_dd=derivatives[-1]
            self.reference_jet_difference=np.stack([derivatives[1][j]-derivatives[0][j] for j in (0,1)])
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
    def spatial_jets(self,q):
        if not self.reference_jets:return self.op.derivatives(q,PARITY)
        d,dd=self.op.derivatives(q-self.initial,PARITY)
        return d+self.reference_d,dd+self.reference_dd
    def advection_jets(self,q):
        if not self.reference_jets:return self.op.advection_derivative(q,PARITY,q[:,9])
        return self.op.advection_derivative(q-self.initial,PARITY,q[:,9])+self.reference_d
    def rhs(self,state,record=False):
        started=time.perf_counter();self.attempted_state=state.copy();q=self.materialize(state)
        d,dd=self.spatial_jets(q)
        R,S,traces,diag=self.kernel.parts(self.r,q,d,dd,self.source)
        rate=np.zeros_like(q);rate[:-1]=R[:-1]+S[:-1]/self.omega[:-1,:1]
        normal=self.normal@(S[-7:]-S[-1])
        if self.normal_policy=='analytic':normal=self.kernel.analytic_normal(q[-1],d[-1],dd[-1])
        rate[-1],dependent=self.kernel.scri(q[-1],d[-1],dd[-1],normal,self.dr[-1],self.penalty)
        q[-1,11]=dependent[0]
        adv=self.advection_jets(q)
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
        tangent_g=rate[:,2]+.5*q[:,2]/q[:,1]*rate[:,1]
        tangent_a=rate[:,4]+.5*q[:,2]/q[:,1]*rate[:,3]-.75*q[:,3]*q[:,2]/q[:,1]**2*rate[:,1]
        self.last=dict(tangency=np.column_stack((tangent_g,tangent_a)),q=q,d=d,dd=dd,diag=diag,pole=S[-1],trace=traces[-1],
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
          alpha_min=float(min(x['q'][:,8])),chi_min=float(min(x['q'][:,0])),rhs_seconds=self.cumulative_rhs_seconds)
        for lo,hi,label in [(.65,.995,'exterior'),(.8,.9,'join'),(.9,.97,'tail'),(.97,.995,'scri_collar')]:
            mask=(self.r>=lo)&(self.r<=hi)
            for f,name in enumerate(('H','M','Theta','Z')):
                out[label+'_'+name]=float(np.sqrt(np.sum(w[mask]*diag[mask,f]**2)/sum(w[mask]))) if np.any(mask) else None
        out['omitted_tangential_rate_max']=float(np.max(abs(x['tangency'])))
        out['omitted_tangential_rate_scri']=float(np.max(abs(x['tangency'][-1])))
        physical=physical_constraints(self.r,x['q'],x['d'],x['dd'],self.omega)
        full_d,full_dd=self.op.derivatives(x['q'],PARITY)
        nodal=physical_constraints(self.r,x['q'],full_d,full_dd,self.omega)
        mask=(self.r[:-1]>=.65)&(self.r[:-1]<=.995)
        for f,name in enumerate(('H','M','Theta','Z')):
            out['physical_'+name]=float(np.sqrt(np.average(physical[mask,f]**2,weights=w[:-1][mask])))
            out['nodal_'+name]=float(np.sqrt(np.average(nodal[mask,f]**2,weights=w[:-1][mask])))
        x['physical_constraints']=physical;x['nodal_constraints']=nodal
        expansion=diag[:,9];roots=np.flatnonzero((expansion[:-1]<0)&(expansion[1:]>=0))
        out['horizon_accepted']=bool(len(roots));out['horizon_mass']=None;out['horizon_r']=None
        if len(roots):
            j=roots[-1];fraction=-expansion[j]/(expansion[j+1]-expansion[j])
            out['horizon_r']=float(self.r[j]+fraction*(self.r[j+1]-self.r[j]))
            out['horizon_mass']=float(diag[j,8]+fraction*(diag[j+1,8]-diag[j,8]))
        out['lapse_gradient_max']=float(max(abs(x['d'][:,8])))
        out['shift_gradient_max']=float(max(abs(x['d'][:,9])))
        return out


class ChartEvolution(Evolution):
    """Ten independent variables; dependent tensor components are never advanced.

    Exact product-rule jets remove the spurious free algebraic modes. Hamiltonian,
    momentum, Theta and the evolved-Lambda constraint are not altered or solved.
    """
    independent=np.array([0,1,3,5,6,7,8,9,10,11])
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.initial_full=self.initial.copy()
        self.initial=self.initial[:,self.independent].copy()
    def materialize(self,state):
        full,_,_=self.kernel.expand_chart(state)
        return super().materialize(full)
    def spatial_jets(self,q):
        values=q[:,self.independent]
        if self.reference_jets:values=values-self.initial
        d,dd=self.op.derivatives(values,PARITY[self.independent])
        if self.reference_jets:d+=self.reference_d[:,self.independent];dd+=self.reference_dd[:,self.independent]
        _,D,DD=self.kernel.expand_chart(q[:,self.independent],d,dd)
        return D,DD
    def advection_jets(self,q):
        values=q[:,self.independent]
        if self.reference_jets:values=values-self.initial
        d=self.op.advection_derivative(values,PARITY[self.independent],q[:,9])
        if self.reference_jets:d+=self.reference_d[:,self.independent]
        _,D,_=self.kernel.expand_chart(q[:,self.independent],d)
        return D
    def rhs(self,state,record=False):
        return super().rhs(state,record)[:,self.independent]
    def step(self,q,dt):
        return super().step(q,dt)[:,self.independent]
