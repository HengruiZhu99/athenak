"""Spatial/RK policies around the C++ current-formulation continuum kernels.

Seven-point polynomial radial derivatives and sign-selected biased advection.
Puncture runs use one-sided stencils, never a fictitious regular-center parity.
No stationary-background subtraction and no elliptic solve.
"""
import sys,time
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from numerics import RadialOperators,weights,puncture_dissipation
from kernel import Kernel,PARITY
from physical_constraints import evaluate as physical_constraints

class Evolution:
    def __init__(self,kernel,n,grading=0.,penalty=.6510416666660156,ko=.5,floors=False,normal_policy="sampled",reference_jets=False,reference_step=5e-4,reference_tail="numerical",grid_map="quadratic"):
        self.kernel=kernel;self.n=n;self.penalty=penalty;self.ko=ko;self.floors=floors;self.normal_policy=normal_policy
        x=(np.arange(n)+.5)/(n-.5)
        # Monotone quadratic map: positive grading refines scri, negative grading
        # refines the puncture chart. dr/dx >= 1-|grading| > 0.
        if not -1<grading<1:raise ValueError('grading must lie in (-1,1)')
        if grid_map not in ("quadratic","two-end"):raise ValueError("unknown radial map")
        shape=x*(1-x) if grid_map=="quadratic" else x*(1-x)*(1-2*x)
        # For two-end: dr/dx=1+g*(1-6*x+6*x²), positive for |g|<1.
        # Negative g refines both ends without imposing puncture parity.
        self.r=x+grading*shape;self.r[-1]=1
        self.dr=np.r_[np.diff(self.r),self.r[-1]-self.r[-2]]
        self.h=float(min(self.dr));self.op=RadialOperators(self.r,width=7,puncture=True,subtract_local=True)
        self.initial,self.source,self.omega=kernel.initial(self.r)
        self.reference_jets=reference_jets;self.reference_jet_difference=None;self.reference_tail_correction=None
        if reference_tail not in ("numerical","scri","outer"):raise ValueError("unknown reference tail policy")
        if reference_tail!="numerical" and not reference_jets:raise ValueError("analytic reference tail requires reference jets")
        if reference_jets:
            derivatives=[]
            if reference_step<=0:raise ValueError("reference step must be positive")
            for step in (2*reference_step,reference_step):
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
            if reference_tail!="numerical":
                mask=self.r==1 if reference_tail=="scri" else self.r>=max(kernel.config[5],kernel.config[12])
                d,dd=kernel.outer_reference_jets(self.r[mask])
                self.reference_tail_correction=np.zeros((2,self.n,12))
                self.reference_tail_correction[0,mask]=d-self.reference_d[mask]
                self.reference_tail_correction[1,mask]=dd-self.reference_dd[mask]
                self.reference_d[mask]=d;self.reference_dd[mask]=dd
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
        if not getattr(self,"evolve_boundary_driver",False):
            q[-1,11]=dependent[0]
            # Auxiliary advection below already uses the materialized trace.
            # Refresh diagnostic centered D jets to describe that same state.
            values=q[:,11]-self.initial[:,-1] if self.reference_jets else q[:,11]
            d[:,11],dd[:,11]=self.op.derivatives(values,PARITY[11])
            if self.reference_jets:
                d[:,11]+=self.reference_d[:,11];dd[:,11]+=self.reference_dd[:,11]
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
        if getattr(self,"evolve_boundary_driver",False):rate[-1,11]=auxiliary[-1]
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
          dependent_D=float(x['dependent_D']),dependent_D_mismatch=float(x['dependent_D']-x['q'][-1,11]),auxiliary_trace_pde=float(x['auxiliary_trace_pde']),
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


class ReferenceChiEvolution(ChartEvolution):
    """Differentiate chi/r^p and recover physical jets in shared Cartesian C++.

    The stored RK values and KO remain physical chi, isolating this spatial
    representation. Fixed linear scaling of the stored RK coordinate would be
    algebraically equivalent. There is no floor or positivity guarantee.
    """
    def __init__(self,*args,chi_reference_power=2,**kwargs):
        super().__init__(*args,**kwargs);self.chi_reference_power=chi_reference_power
        if chi_reference_power not in (2,4):raise ValueError('unsupported chi reference power')
        self.scaled_initial,self.scaled_d,self.scaled_dd=self.kernel.reference_chi(self.r,self.initial,chi_reference_power,
            self.reference_d[:,self.independent] if self.reference_jets else None,
            self.reference_dd[:,self.independent] if self.reference_jets else None,inverse=True)
    def scaled_values(self,q):
        return self.kernel.reference_chi(self.r,q[:,self.independent],self.chi_reference_power,inverse=True)[0]
    def spatial_jets(self,q):
        scaled=self.scaled_values(q);values=scaled-self.scaled_initial if self.reference_jets else scaled
        d,dd=self.op.derivatives(values,PARITY[self.independent])
        if self.reference_jets:d+=self.scaled_d;dd+=self.scaled_dd
        Q,D,DD=self.kernel.reference_chi(self.r,scaled,self.chi_reference_power,d,dd)
        return self.kernel.expand_chart(Q,D,DD)[1:]
    def advection_jets(self,q):
        scaled=self.scaled_values(q);values=scaled-self.scaled_initial if self.reference_jets else scaled
        d=self.op.advection_derivative(values,PARITY[self.independent],q[:,9])
        if self.reference_jets:d+=self.scaled_d
        Q,D,_=self.kernel.reference_chi(self.r,scaled,self.chi_reference_power,d)
        return self.kernel.expand_chart(Q,D)[1]


class DerivativeBoundaryEvolution(ChartEvolution):
    """Experimental normal-derivative closure with the shared D PDE at scri.

    The last three gauge derivative rows use a degree-seven Hermite polynomial
    through the last seven values plus the normal derivative. No nodal data are
    smoothed. Its correction vanishes at every interpolation node. Reference
    jets are retained: the correction acts on their residual compatibility.
    This is a spatial boundary policy, not a boundary well-posedness proof.
    """
    evolve_boundary_driver=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.normal_policy!='analytic':raise ValueError('derivative boundary requires analytic pole normal')
        z=(self.r[-7:]-1)/self.dr[-1]
        poly=np.polynomial.Polynomial.fromroots(z)
        poly=poly/poly.deriv()(0)
        self.hermite_d=poly.deriv()(z[-3:])
        self.hermite_dd=poly.deriv(2)(z[-3:])/self.dr[-1]
    def spatial_jets(self,q):
        d,dd=super().spatial_jets(q)
        def condition(derivative):
            normal=self.kernel.analytic_normal(q[-1],derivative,dd[-1])
            _,dep=self.kernel.scri(q[-1],derivative,dd[-1],normal,self.dr[-1],self.penalty)
            return np.array([dep[1],dep[0]-q[-1,11]])
        base=condition(d[-1]);columns=[]
        for f in (8,9):
            perturbed=d[-1].copy();perturbed[f]+=1
            columns.append(condition(perturbed)-base)
        matrix=np.array(columns).T
        correction=np.linalg.solve(matrix,-base)
        d[-3:,8:10]+=self.hermite_d[:,None]*correction
        dd[-3:,8:10]+=self.hermite_dd[:,None]*correction
        residual=condition(d[-1])
        if np.max(abs(residual))>1e-8*(1+np.max(abs(base))):
            raise FloatingPointError('nonlinear boundary derivative compatibility')
        self.boundary_matrix_condition=float(np.linalg.cond(matrix))
        self.boundary_derivative_correction=correction
        self.boundary_gauge_jets=d[-3:,8:10].copy()
        return d,dd
    def advection_jets(self,q):
        d=super().advection_jets(q)
        d[-3:,8:10]=self.boundary_gauge_jets
        return d


def make_evolution(kernel,cfg):
    power=cfg.get('chi_reference_power',0)
    if power and (not cfg.get('algebraic_chart',True) or cfg.get('derivative_boundary',False)):
        raise ValueError('reference chi needs the algebraic chart and standard boundary')
    cls=ReferenceChiEvolution if power else DerivativeBoundaryEvolution if cfg.get('derivative_boundary',False) else ChartEvolution if cfg.get('algebraic_chart',True) else Evolution
    extra={'chi_reference_power':power} if power else {}
    return cls(kernel,cfg.get('n',64),cfg.get('grading',0.),cfg.get('penalty',.6510416666660156),cfg.get('ko',.5),cfg.get('floors',False),cfg.get('normal_policy','analytic'),cfg.get('reference_jets',True),cfg.get('reference_step',.000125),cfg.get('reference_tail','numerical'),cfg.get('grid_map','quadratic'),**extra)
