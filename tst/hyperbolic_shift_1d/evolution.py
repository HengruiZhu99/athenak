"""Bounded regular-center prototype using the complete spherical tensor kernel.

No metric freezing, connection replacement, or bulk constraint projection.
Puncture runs are deliberately gated on regular-center qualification.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from kernel import Kernel, PARITY
from numerics import RadialOperators, puncture_dissipation
from telegraph import window
from gauge import lapse_coefficient, restoring_coefficient, collapsed_background_correction
from elliptic import solve, connection
from boundary import normal_shift, normal_rate
from principal import coefficient, speeds
from shift_source import background
from metric_driver import transport_rate
from gamma_driver import parameters as gamma_parameters
from constraint_damping import correction as damping_correction

class Evolution:
    def __init__(self,n,library,ratio=2.,damping=1.,ko=.5,kind='separated',mode='hyperbolic',restoring=0.,inner_shift=0.,lapse_floor=1e-10,shift_source='reference',connection_driver='evolved',shift_gauge='separated',gamma_coefficient=1.,inner_z4_damping=0.):
        self.r=(np.arange(n)+.5)/(n-.5);self.h=1/(n-.5)
        self.O=(1-self.r**2)/2;self.A=(1+self.r**2)/2
        self.op=RadialOperators(self.r,width=7,subtract_local=True)
        self.kernel=Kernel(library);self.ratio=ratio;self.damping=damping
        self.ko=ko;self.kind=kind;self.mode=mode
        self.restoring=restoring
        self.inner_shift=inner_shift
        self.shift_source=shift_source
        self.connection_driver=connection_driver
        if not np.isfinite(inner_z4_damping) or inner_z4_damping<0:raise ValueError('Invalid inner Z4 damping')
        self.inner_z4_damping=inner_z4_damping
        self.shift_gauge=shift_gauge;self.gamma_coefficient=gamma_coefficient
        if shift_gauge not in ['separated','moving-puncture']:raise ValueError('Unknown shift gauge')
        if shift_gauge=='moving-puncture':
            if shift_source!='reference' or connection_driver!='evolved' or inner_shift!=0 or mode!='hyperbolic':
                raise ValueError('Moving-puncture candidate requires the reference source and evolved connection')
            gamma_parameters(self.r,np.ones(n),restoring,gamma_coefficient)
        if not np.isfinite(lapse_floor) or lapse_floor<0:raise ValueError('Invalid physical lapse floor')
        self.lapse_floor=lapse_floor
        self.w,self.wp=window(self.r,width=.24,inner_profile='af-regular')
        self.xi=restoring_coefficient(self.r,1.,1.,profile='physical-log')
        self.parity=np.r_[PARITY,-1,-1]
        self.stats=dict(calls=0,max_null=0.,max_relative_null=0.,max_driver_boundary_defect=0.,
                        max_pole=0.,max_trace=0.,max_speed=0.,max_floor_radius=0.,floor_calls=0)

    def materialize(self,state):
        q=state.copy();alpha=self.A+q[:,6];chi=1+q[:,0];g=1+q[:,1]
        floor=self.lapse_floor*self.O;active=(alpha<floor)&(self.r<1)
        if np.any(active):
            q[active,6]=floor[active]-self.A[active];alpha=self.A+q[:,6]
            self.stats['max_floor_radius']=max(self.stats['max_floor_radius'],float(max(self.r[active])))
            self.stats['floor_calls']+=1
        if np.any(chi<=0) or np.any(g<=0) or np.any(alpha<=0) or not np.isfinite(q).all():
            raise FloatingPointError('Nonpositive/nonfinite full variable-metric stage')
        q[-1,7]=1+normal_shift(alpha[-1],chi[-1],g[-1])
        return q

    def rhs(self,state,return_state=False):
        q=self.materialize(state);r=self.r
        d,dd=self.op.derivatives(q,self.parity)
        alpha=self.A+q[:,6];chi=1+q[:,0];g=1+q[:,1]
        if self.mode=='elliptic':
            beta,db,ddb,_=solve(self.op,g,d[:,1],dd[:,1],alpha,r+d[:,6],q[:,2],d[:,2],q[-1,7]-1)
            q[:,7]=beta+r
        else:beta=-r+q[:,7];db=-1+d[:,7];ddb=dd[:,7]
        R,S,diag=self.kernel.parts(r,q[:,:8],d[:,:8],dd[:,:8],beta,db,ddb)
        if self.inner_z4_damping:
            S+=damping_correction(r,alpha,q[:,4],q[:,5]-connection(r,g,d[:,1]),self.w,self.inner_z4_damping)
        # Identical lapse sources to the successful baseline; no reference-RHS subtraction.
        S[:,6]-=2*self.xi*alpha*q[:,6]
        S[:,6]+=collapsed_background_correction(r,q[:,6])
        eta=1.0001*alpha/(alpha+1e-4*self.A)
        R[:,6]+=eta*chi*(self.w*d[:,8]+(self.wp+2*self.w/r)*q[:,8])
        rate=np.zeros_like(q);rate[:-1,:8]=R[:-1]+S[:-1]/self.O[:-1,None]
        ds=self.op.endpoint_first(S)
        rate[-1,:8]=R[-1]-ds
        trace=np.zeros(8)
        trace[0]=chi[-1]-g[-1]
        trace[2]=q[-1,2]-d[-1,1]/6+2*d[-1,0]/3
        trace[3:5]=q[-1,3:5]
        trace[5]=q[-1,5]-connection(1.,g[-1],d[-1,1])
        rate[-1,:8]+=4/self.h*S[-1]-8/self.h*trace
        F=lapse_coefficient(r,alpha)
        mu=coefficient(alpha,chi,g,F,eta,self.w,.005,.02,self.ratio,self.kind,self.inner_shift,self.O)
        xi_shift=self.restoring
        if self.shift_gauge=='moving-puncture':mu,xi_shift=gamma_parameters(r,mu,self.restoring,self.gamma_coefficient)
        rate[:,8]=beta*d[:,8]+(.005*self.w*d[:,6]-q[:,8])/.02
        source,target_correction,target_prime=background(r,alpha,q[:,6],d[:,6],self.shift_source,self.w,self.wp)
        if self.mode=='hyperbolic':
            # D is the auxiliary Gamma driver, not the independent Z4 connection.
            rate[:,7]=beta*db+source+mu*q[:,9]
            rate[:-1,7]-=np.broadcast_to(xi_shift,r.shape)[:-1]*(q[:-1,7]+target_correction[:-1])/self.O[:-1]
        adv=self.op.advection_derivative(q,self.parity,beta)
        rate[:,:9]+=beta[:,None]*(adv[:,:9]-d[:,:9])
        # Baseline KO8 coefficient and taper. No scri filtering.
        diss=puncture_dissipation(q,r,self.h,self.ko,.8,order=8,include_first=True)
        rate[:,:9]+=diss[:,:9]
        rate[-1,6]=0.
        bt=normal_rate(alpha[-1],chi[-1],g[-1],0.,rate[-1,0],rate[-1,1])
        if self.mode=='hyperbolic':
            # The single boundary datum beta_n=-alpha implies this driver trace.
            # Do not prescribe D independently. Use the same beta derivative as
            # in the advective driver equation when solving its compatibility.
            db_driver=-1+adv[-1,7]
            source_limit=float(np.broadcast_to(xi_shift,r.shape)[-1])*(d[-1,7]+target_prime[-1])
            q[-1,9]=(bt-beta[-1]*db_driver-source[-1]-source_limit)/mu[-1]
            aD=self.op.advection_derivative(q[:,9,None],np.array([-1]),beta)[:,0]
            drive=rate[:,5]-beta*adv[:,5]
            if self.connection_driver=='metric':drive=transport_rate(r,g,d[:,1],alpha,r+d[:,6],q[:,2],d[:,2],beta,db,ddb)
            rate[:,9]=beta*aD+drive-self.damping*q[:,9]
            rate[:,9]+=puncture_dissipation(q[:,9,None],r,self.h,self.ko,.8,order=8,include_first=True)[:,0]
            rate[-1,9]=0. # eliminated storage slot; rematerialized at each stage
            defect=abs(bt-(beta[-1]*db_driver+source[-1]+mu[-1]*q[-1,9]+source_limit))
        else:rate[:,9]=0.;defect=0.
        rate[-1,7]=bt
        null=chi[-1]/g[-1]-(beta[-1]/alpha[-1])**2
        maxspeed=np.max(abs(beta)+np.maximum(np.sqrt(chi*F/g+eta*chi*.005*self.w**2/.02),np.sqrt(4*mu/(3*g))))
        raw=R[:-1]+S[:-1]/self.O[:-1,None]
        self.last=dict(diag=diag,beta=beta,alpha=alpha,chi=chi,g=g,pole=S[-1],trace=trace,
                       raw_clock_rate=float(R[-1,6]-ds[6]),
                       original_equation_rate_difference=float(np.max(abs(rate[:-1,:7]-raw[:,:7]))))
        st=self.stats;st['calls']+=1
        for key,val in [('max_null',abs(null)),('max_relative_null',abs(null)/(chi[-1]/g[-1])),
                        ('max_driver_boundary_defect',defect),('max_pole',max(abs(S[-1]))),
                        ('max_trace',max(abs(trace))),('max_speed',maxspeed)]:st[key]=max(st[key],float(val))
        st['max_shift_pole']=max(st.get('max_shift_pole',0.),abs(float(self.restoring*(q[-1,7]+target_correction[-1]))))
        if not np.isfinite(rate).all():raise FloatingPointError('Nonfinite RHS')
        return (rate,q) if return_state else rate

    def step(self,q,dt):
        a=self.rhs(q);b=self.rhs(q+dt*a/2);c=self.rhs(q+dt*b/2);d=self.rhs(q+dt*c)
        increment=dt*(a+2*b+2*c+d)/6
        if not hasattr(self,'compensation'):self.compensation=np.zeros_like(q)
        adjusted=increment-self.compensation;accepted=q+adjusted
        self.compensation=(accepted-q)-adjusted
        _,accepted=self.rhs(accepted,return_state=True)
        self.compensation[-1,[6,7,9]]=0.
        return accepted

    def report(self,q,t):
        self.rhs(q);x=self.last;w=self.r**2
        return dict(t=t,**{k:float(np.sqrt(np.sum(w*x['diag'][:,i]**2)/sum(w))) for i,k in enumerate(['H','M','Theta','Z'])},
                    pole=float(max(abs(x['pole']))),trace=float(max(abs(x['trace']))),
                    alpha_min=float(min(x['alpha'])),g_min=float(min(x['g'])),
                    metric_change=float(max(abs(q[:,1]))),**self.stats,
                    original_equation_rate_difference=x['original_equation_rate_difference'])
