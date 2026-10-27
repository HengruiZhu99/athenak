"""Positive AF puncture chart with an evolved asymptotic-infinity endpoint.

All six physical field functions remain independent. The r=0 point holds
finite asymptotic coefficients, not a regular center or a physical metric.
"""
import time
from types import SimpleNamespace
import numpy as np
from evolve import Evolution
from wormhole_data import Jet
from numerics import puncture_dissipation,weights
from gauge import rational_correction,af_slicing_strength,collapsed_background_correction,blended_correction,restoring_coefficient
import g1_rhs,scri_limits


class PunctureChart(Evolution):
    def __init__(self,args):
        super().__init__(args)
        self.positive_stencil=getattr(args,'puncture_stencil','coefficient')=='positive'
        if self.positive_stencil:
            if self.r[0]!=0 or getattr(args,'lapse_background_source','reference')!='collapsed-inner':raise ValueError('Positive-node closure requires the full AF chart and collapsed-inner source')
            # The asymptotic coefficient is not a resolved nodal sample once the
            # inward AF layer falls below the grid. Keep it out of bulk stencils.
            op=self.op;offset=op.width
            for i in range(1,len(self.r)):
                start=max(int(op.index[i,0]),offset+1);idx=np.arange(start,start+op.width)
                op.index[i]=idx;op.d1[i]=weights(op.ext[idx],self.r[i],1);op.d2[i]=weights(op.ext[idx],self.r[i],2)
                for direction in range(2):
                    start=max(int(op.aindex[direction,i,0]),offset+1);idx=np.arange(start,start+op.width)
                    op.aindex[direction,i]=idx;op.ad1[direction,i]=weights(op.ext[idx],self.r[i],1)
        self.inner_total_radius=getattr(args,'inner_total_derivatives_radius',0.)
        if not 0<=self.inner_total_radius<1 or (self.inner_total_radius>0 and not self.positive_stencil):raise ValueError('Inner total derivatives require positive-node closure and radius in [0,1)')
        self.refs={key:[] for key in ['chi','w','alpha','c','u','gap']}
        for value in self.r:
            r=Jet(float(value),1);O=(1-r*r)/2;A=(1+r*r)/2
            chi=r**4/(r**4+O**4);w=r*r/(r*r+O**4);gap=O**4/(r*r+O**4)
            fields=dict(chi=chi,w=w,alpha=A*w,c=-O**4/(r**4+O**4),u=-A*gap,gap=gap)
            for key,z in fields.items():self.refs[key].append([z.v,z.d,z.dd])
        self.positive_stencil=getattr(args,'puncture_stencil','coefficient')=='positive'
        if self.positive_stencil:
            if self.r[0]!=0 or getattr(args,'lapse_background_source','reference')!='collapsed-inner':raise ValueError('Positive-node closure requires the full AF chart and collapsed-inner source')
            # The asymptotic coefficient is not a resolved nodal sample once the
            # inward AF layer falls below the grid. Keep it out of bulk stencils.
            op=self.op;offset=op.width
            for i in range(1,len(self.r)):
                start=max(int(op.index[i,0]),offset+1);idx=np.arange(start,start+op.width)
                op.index[i]=idx;op.d1[i]=weights(op.ext[idx],self.r[i],1);op.d2[i]=weights(op.ext[idx],self.r[i],2)
                for direction in range(2):
                    start=max(int(op.aindex[direction,i,0]),offset+1);idx=np.arange(start,start+op.width)
                    op.aindex[direction,i]=idx;op.ad1[direction,i]=weights(op.ext[idx],self.r[i],1)
        self.refs={key:np.asarray(value) for key,value in self.refs.items()}
        self.stage_metrics={};self.stage_calls=0

    def reconstruct(self,state,d=None,dd=None):
        with np.errstate(over='raise',invalid='raise'):
            ex=np.zeros(len(state));ea=np.zeros(len(state));first=int(self.positive_stencil)
            ex[first:]=np.exp(state[first:,0]);ea[first:]=np.exp(state[first:,6])
        if not np.all(np.isfinite(ex)) or not np.all(np.isfinite(ea)):raise FloatingPointError('Nonfinite puncture logarithm')
        cr=self.refs['chi'];ar=self.refs['alpha'];w=self.refs['w'];gap=self.refs['gap']
        chi=cr[:,0]*ex;alpha=ar[:,0]*ea;khat=w[:,0]*(state[:,3]-3)
        q=np.zeros_like(state)
        smc=np.empty(len(state));sma=np.empty(len(state));smc[:first]=-1;sma[:first]=-1;smc[first:]=np.expm1(state[first:,0]);sma[first:]=np.expm1(state[first:,6])
        q[:,0]=self.refs['c'][:,0]+cr[:,0]*smc
        q[:,2]=w[:,0]*state[:,2];q[:,3]=3*gap[:,0]+w[:,0]*state[:,3]
        q[:,4]=w[:,0]*state[:,4];q[:,5]=state[:,5]
        q[:,6]=self.refs['u'][:,0]+ar[:,0]*sma
        if d is None:return q,chi,alpha,khat
        dp=np.zeros_like(state);ddp=np.zeros_like(state)
        for f,ref,exponential in [(0,cr,ex),(6,ar,ea)]:
            dp[:,f]=exponential*(ref[:,1]+ref[:,0]*d[:,f])
            ddp[:,f]=exponential*(ref[:,2]+2*ref[:,1]*d[:,f]+ref[:,0]*(dd[:,f]+d[:,f]**2))
        # Differentiate the deviation directly: total-alpha minus A loses tiny
        # deviations in the regular scri collar.
        small=sma;uref=self.refs['u']
        dp[:,6]=uref[:,1]+ar[:,1]*small+ar[:,0]*ea*d[:,6]
        ddp[:,6]=uref[:,2]+ar[:,2]*small+ea*(2*ar[:,1]*d[:,6]+ar[:,0]*(dd[:,6]+d[:,6]**2))
        for f in [2,3,4]:
            value=state[:,f]-(3 if f==3 else 0)
            dp[:,f]=w[:,1]*value+w[:,0]*d[:,f]
            ddp[:,f]=w[:,2]*value+2*w[:,1]*d[:,f]+w[:,0]*dd[:,f]
        dp[:,5]=d[:,5];ddp[:,5]=dd[:,5]
        if self.inner_total_radius>0:
            totals=q.copy();totals[:,0]=chi;totals[:,3]=khat;totals[:,6]=alpha
            direct,direct2=self.op.derivatives(totals,np.ones(8));direct[:,6]-=self.r;direct2[:,6]-=1
            y=np.clip(2*(1-self.r/self.inner_total_radius),0,1);blend=(y**3*(10+y*(-15+6*y)))[:,None];blend[0]=0
            fields=[0,2,3,4,6];dp[:,fields]=(1-blend)*dp[:,fields]+blend*direct[:,fields];ddp[:,fields]=(1-blend)*ddp[:,fields]+blend*direct2[:,fields]
        return q,chi,alpha,khat,dp,ddp

    def to_physical(self,state):return self.reconstruct(state)[0]

    def initial_wormhole(self,initial):
        if self.r[0]!=0:raise ValueError('Finite-cutoff AF charts require an explicit domain transfer')
        iq,id1,id2,meta=initial.data(self.r[1:]);q=np.vstack([np.array([-1.,0,0,3,0,0,-.5,0]),iq])
        chi=np.r_[0.,[z['chi_total'] for z in meta]]
        alpha=np.zeros_like(self.r);alpha[1:-1]=np.array([z['physical_lapse'] for z in meta[:-1]])*self.omega[1:-1];alpha[-1]=1.
        state=np.zeros_like(q);positive=self.r>0;inner=positive&(self.r<.5);outer=self.r>=.5
        state[inner,0]=np.log(chi[inner]/self.refs['chi'][inner,0])
        state[outer,0]=np.log1p((q[outer,0]-self.refs['c'][outer,0])/self.refs['chi'][outer,0])
        state[inner,6]=np.log(alpha[inner]/self.refs['alpha'][inner,0])
        state[outer,6]=np.log1p((q[outer,6]-self.refs['u'][outer,0])/self.refs['alpha'][outer,0])
        w=self.refs['w'][:,0]
        state[positive,2]=q[positive,2]/w[positive]
        state[inner,3]=3+(q[inner,3]-3)/w[inner]
        state[outer,3]=(q[outer,3]-3*self.refs['gap'][outer,0])/w[outer]
        state[0,0]=np.log(4/(initial.mass**4*initial.k**2))
        state[0,6]=np.log(1/(4*initial.mass**2*initial.k**2));state[0,3]=3
        endpoint_dd=np.zeros(8);endpoint_dd[6]=4/(initial.mass**2*initial.k**2)-1
        return state,q,np.vstack([np.zeros(8),id1]),np.vstack([endpoint_dd,id2]),[dict(R=float('inf'),chi_total=0.,branch='AF coefficient endpoint'),*meta]

    def parts(self,state):
        first=int(self.r[0]==0);physical=slice(first,None);rr=self.r[physical]
        d,dd=self.op.derivatives(state,np.ones(8))
        q,chi,alpha,khat,dp,ddp=self.reconstruct(state,d,dd)
        if min(chi[physical])<=0 or min(alpha[physical])<=0:raise FloatingPointError('Nonpositive physical puncture geometry')
        self.geometry_state=q;self.geometry_first=dp;self.geometry_second=ddp
        self.chart_first=d;self.chart_second=dd;self.total_chi=chi;self.total_alpha=alpha;self.total_khat=khat
        start=time.perf_counter();integrand=np.zeros(len(self.r));integrand[physical]=1.5*alpha[physical]*q[physical,2]/rr
        integral=self.op.integral(integrand);integral_seconds=time.perf_counter()-start
        sub=SimpleNamespace(r=rr,integral=lambda _:integral[physical])
        timing={}
        R,S,diag,b,db,ddb=g1_rhs.parts(sub,q[physical],dp[physical],ddp[physical],self.args.kappa1,self.args.kappa2,
            self.args.theta=='appendix-c',self.args.slicing,0.,timing,
            total_chi=chi[physical],total_alpha=alpha[physical],total_khat=khat[physical],puncture_cancellation=True)
        if getattr(self.args,'lapse_restoring_profile','constant')=='physical-local':
            self.restoring_strength=restoring_coefficient(self.r,self.args.lapse_restoring,self.args.slicing,
                getattr(self.args,'slicing_family','linear'),getattr(self.args,'slicing_scale',.1),'physical-local',alpha)
        strength=self.restoring_strength if np.ndim(self.restoring_strength)==0 else self.restoring_strength[physical]
        S[:,6]-=2*strength*alpha[physical]*q[physical,6]
        if getattr(self.args,'slicing_family','linear')=='rational':S[:,6]+=rational_correction(rr,alpha[physical],q[physical,6],q[physical,3],self.args.slicing,self.args.slicing_scale)
        if getattr(self.args,'slicing_family','linear')=='blended-log':S[:,6]+=blended_correction(rr,alpha[physical],q[physical,3],self.args.slicing)
        if getattr(self.args,'lapse_background_source','reference')=='collapsed-inner':S[:,6]+=collapsed_background_correction(rr,q[physical,6])
        if getattr(self.args,'damping_form','constant')=='reference-lapse':
            D=-self.args.kappa1*q[physical,6]
            S[:,3]+=D*(1-self.args.kappa2)*q[physical,4]
            S[:,4]-=D*(2+self.args.kappa2)*q[physical,4];S[:,5]-=D*q[physical,5]
        beta=b;v=timing['beta_deviation'];self.mass_numerator=timing['mass_numerator']
        cS=q[-1,0];uS=q[-1,6]
        outer=-(cS+(2*uS+uS*uS)*chi[-1])/(1+alpha[-1]*np.sqrt(chi[-1]))
        self.b1=-1+outer-integral[-1]
        if first:
            R=np.vstack([np.zeros(8),R]);S=np.vstack([np.zeros(8),S]);diag=np.vstack([np.zeros(12),diag])
            beta=np.r_[0.,b];v=np.r_[0.,v];self.mass_numerator=np.r_[0.,self.mass_numerator]
            diag[0,7]=0.;diag[0,8]=0.;diag[0,9]=1.;diag[0,3]=state[0,5]/2
            if self.positive_stencil:diag[0,10]=-.25*d[0,0]*np.exp(-state[0,0]/2)
            else:
                X0=16*np.exp(state[0,0]);diag[0,10]=-d[0,0]/np.sqrt(X0)
            diag[0,11]=0.
        else:
            from gauge import lapse_coefficient
            F=lapse_coefficient(rr[0],alpha[0],self.args.slicing,getattr(self.args,'slicing_family','linear'),getattr(self.args,'slicing_scale',.1))
            light=-beta[0]+alpha[0]*np.sqrt(chi[0]);gauge=-beta[0]+np.sqrt(F*chi[0])
            metrics=self.stage_metrics
            metrics['max_inner_characteristic']=max(metrics.get('max_inner_characteristic',-float('inf')),float(max(light,gauge)))
            if max(light,gauge)>0:raise FloatingPointError('Transferred inner boundary has an incoming physical or gauge characteristic')
        self.shift_deviation=v;self.origin_shift_slope=self.b1
        self.calls+=1;self.solve_seconds+=integral_seconds+timing['elliptic_seconds']
        fd,fdd=self.op.derivatives(v,-1)
        er=4/3*(fdd[physical]+2*(fd[physical]-v[physical]/rr)/rr)-2*((rr+dp[physical,6])*q[physical,2]+alpha[physical]*dp[physical,2])-6*alpha[physical]*q[physical,2]/rr
        self.elliptic_differential_residual=float(max(abs(er)))
        self.regularity=np.array([q[-1,0],0.,q[-1,2]+2*dp[-1,0]/3,q[-1,3],q[-1,4],q[-1,5],0.,0.])
        return R,S,diag,beta

    def rhs(self,state,accepted=False):
        start=time.perf_counter();R,S,diag,beta=self.parts(state)
        first=int(self.r[0]==0)
        rate=np.zeros_like(state);rate[first:-1]=R[first:-1]+S[first:-1]/self.omega[first:-1,None]
        if self.args.endpoint=='analytic-g1':
            ds,_=scri_limits.derivatives(self.geometry_state[-1],self.geometry_first[-1],self.geometry_second[-1],
                self.shift_deviation[-1],self.args.kappa1,self.args.kappa2,self.args.theta=='appendix-c',self.args.slicing,self.args.lapse_restoring,restoring_form='multiplicative',damping_form=getattr(self.args,'damping_form','constant'))
        else:ds=self.op.endpoint_first(S)
        rate[-1]=R[-1]-ds
        hs=self.h*self.grid_jacobian[-1]
        rate[-1]+=self.args.scri_sat/hs*S[-1]-self.args.regularity_sat/hs*self.regularity
        rate[-1,6]=0.;rate[:,7]=0.
        result=np.zeros_like(state);positive=self.r>0;w=self.refs['w'][:,0]
        result[positive,0]=rate[positive,0]/self.total_chi[positive]
        result[positive,6]=rate[positive,6]/self.total_alpha[positive]
        for f in [2,3,4]:result[positive,f]=rate[positive,f]/w[positive]
        result[:,5]=rate[:,5]
        if first:
            b1=self.b1;xi=float(self.restoring_strength if np.ndim(self.restoring_strength)==0 else self.restoring_strength[0]);nu=af_slicing_strength(self.args.slicing,getattr(self.args,'slicing_family','linear'),getattr(self.args,'slicing_scale',.1));A0=8*np.exp(state[0,6]) if not self.positive_stencil else None
            result[0,0]=2*b1;result[0,6]=2*b1-6*nu+2*xi
            if getattr(self.args,'lapse_background_source','reference')=='reference':result[0,6]+=2/A0
            result[0,2]=2*b1*state[0,2];result[0,3]=2*b1*(state[0,3]-3)
            result[0,4]=2*b1*state[0,4];result[0,5]=2*b1*state[0,5]
            if getattr(self.args,'damping_form','constant')=='reference-lapse':
                result[0,3]+=self.args.kappa1*(1-self.args.kappa2)*state[0,4]
                result[0,4]-=self.args.kappa1*(2+self.args.kappa2)*state[0,4]
                result[0,5]-=self.args.kappa1*state[0,5]
        if self.args.advection=='upwind':
            correction=self.op.advection_derivative(state,np.ones(8),beta)-self.chart_first
            if self.inner_total_radius>0:
                totals=self.geometry_state.copy();totals[:,0]=self.total_chi;totals[:,3]=self.total_khat;totals[:,6]=self.total_alpha
                direct,_=self.op.derivatives(totals,np.ones(8));direct=self.op.advection_derivative(totals,np.ones(8),beta)-direct
                y=np.clip(2*(1-self.r/self.inner_total_radius),0,1);blend=y**3*(10+y*(-15+6*y));blend[0]=0
                for f,scale in [(0,self.total_chi),(2,w),(3,w),(4,w),(6,self.total_alpha)]:
                    correction[positive,f]=(1-blend[positive])*correction[positive,f]+blend[positive]*direct[positive,f]/scale[positive]
            result+=beta[:,None]*correction
        if getattr(self.args,'puncture_ko',0.):
            measure=self.grid_jacobian if getattr(self.args,'mapped_ko','physical')=='local' else None
            diss=puncture_dissipation(state,self.r,self.h,self.args.puncture_ko,self.args.puncture_ko_radius,inner_width=getattr(self.args,'puncture_ko_inner_width',0.),measure=measure,order=getattr(self.args,'ko_order',6))
            diss/=self.grid_jacobian[:,None]
            if getattr(self.args,'radial_operator','fd')=='sbp63':diss/=self.op.norm[:,None]
            result+=diss
        result[:,1]=0.;result[:,7]=0.;result[-1,6]=0.
        if not np.all(np.isfinite(result)):raise FloatingPointError('Nonfinite AF coefficient rate')
        self.stage_calls+=1;metrics=self.stage_metrics
        maxima=dict(max_scri_pole=float(max(abs(S[-1,:6]))),max_lapse_pole=float(abs(S[-1,6])),
            max_regularity_trace=float(max(abs(self.regularity))),max_scri_null=float(abs(diag[-1,6])),max_scri_clock=float(abs(self.geometry_state[-1,6])))
        from bondi import nonaffinity
        maxima['max_Bondi_nonaffinity']=float(abs(nonaffinity(self.total_chi[-1],self.total_alpha[-1],
            self.geometry_first[-1,0],1+self.geometry_first[-1,6],beta[-1]+1.5*self.total_alpha[-1]*self.geometry_state[-1,2],self.total_chi[-1]*result[-1,0])))
        measure=self.r**2*self.grid_jacobian
        for i,name in enumerate(['H','M','Theta','Z']):maxima['max_'+name+'_rms']=float(np.sqrt(np.sum(measure*diag[:,i]**2)/np.sum(measure)))
        if accepted:
            for i,name in enumerate(['H','M','Theta','Z']):maxima['max_accepted_'+name+'_rms']=maxima['max_'+name+'_rms']
            for i,name in enumerate(['H','M']):
                maxima['max_accepted_'+name+'_bulk']=float(max(abs(diag[self.r<.8,i])))
                maxima['max_accepted_'+name+'_endpoint']=float(max(abs(diag[self.r>.9,i])))
        for key,value in maxima.items():metrics[key]=max(metrics.get(key,0),value)
        for key,value in [('min_chi',float(min(self.total_chi[positive]))),('min_alpha',float(min(self.total_alpha[positive])))] :metrics[key]=min(metrics.get(key,float('inf')),value)
        self.rhs_seconds+=time.perf_counter()-start
        return result

    def report(self,state,t):
        row,b,c=super().report(state,t)
        if self.r[0]>0:
            row['inner_boundary_treatment']='causal finite cutoff in positive AF-reference variables'
            if self.stage_metrics:row['all_RK_stages']=dict(self.stage_metrics)
            return row,b,c
        row.update(AF_log_chi_coefficient=float(state[0,0]),AF_log_lapse_coefficient=float(state[0,6]),
            AF_inner_mass=float(c[0,10]),AF_shift_slope=float(self.b1),inner_boundary_treatment='evolved AF coefficient endpoint')
        if self.stage_metrics:row['all_RK_stages']=dict(self.stage_metrics)
        return row,b,c
