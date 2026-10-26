#!/usr/bin/env python3
"""Experimental actual-scri free-Z4c driver; every failed variant is retained.

Regular-center vacuum/scalar and a separately guarded vacuum puncture path.
"""
import argparse,hashlib,json,os,platform,sys,time,shutil
from pathlib import Path
import numpy as np
from elliptic import solve,residual,connection
from numerics import RadialOperators,weights
from kernel import Kernel,PARITY
from gauge import lapse_coefficient,rational_correction,collapsed_background_correction,blended_correction,restoring_coefficient
from horizon import apparent_horizons
from time_lattice import FixedStepClock


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evolution:
    def __init__(self,args):
        self.args=args;inner=getattr(args,'inner_radius',0.)
        nodal=getattr(args,'origin_grid',False) or getattr(args,'puncture_chart',False) or inner>0
        self.r=np.linspace(inner,1,args.n) if nodal else (np.arange(args.n)+.5)/(args.n-.5)
        self.h=(1-inner)/(args.n-1) if nodal else 1/(args.n-.5);self.op=RadialOperators(self.r,width=args.width,quadrature_width=args.quadrature_width,puncture=getattr(args,'puncture',False),allow_puncture_node=getattr(args,'puncture_chart',False))
        if getattr(args,'radial_operator','fd')=='sbp63':
            from sbp import SBPOperators
            self.op=SBPOperators(self.r,args.width,args.quadrature_width)
        self.op.elliptic_quadrature=getattr(args,'elliptic_quadrature','prefix')
        if getattr(args,'elliptic_method','quadrature')=='discrete-ode':
            from shift_ode import BandedShift
            self.op.shift_ode_solver=BandedShift(self.op,getattr(args,'elliptic_refinements',2),getattr(args,'elliptic_bias','centered'))
        self.kernel=Kernel(args.library);self.calls=0;self.solve_seconds=0.
        self.stage_metrics={};self.stage_calls=0
        self.restoring_strength=restoring_coefficient(self.r,args.lapse_restoring,args.slicing,
            getattr(args,'slicing_family','linear'),getattr(args,'slicing_scale',.1),getattr(args,'lapse_restoring_profile','constant'))
        self.lapse_source=np.zeros(args.n)
        if getattr(args,'lapse_reference','minkowski')=='CMC-trumpet':
            from trumpet_data import CMCTrumpet
            self.lapse_source=CMCTrumpet(args.mass,args.initial_quadrature_order).lapse_source(
                self.r,self.restoring_strength,getattr(args,'lapse_restoring_form','symmetric'))
        self.alpha_profile=None
        self.metric_scale=None
        self.trumpet_metric=getattr(args,'metric_variable','deviation')=='log-trumpet'
        if getattr(args,'metric_variable','deviation')=='log':self.metric_scale=(np.ones(args.n),np.zeros(args.n))
        if getattr(args,'metric_variable','deviation')=='log-trumpet':
            A=(1+self.r*self.r)/2;O=(1-self.r*self.r)/2
            self.metric_scale=(self.r*self.r/A**2,-O**2/A**2)
        if getattr(args,'lapse_variable','deviation')=='log-trumpet':
            from trumpet_data import CMCTrumpet
            ref,rd,rdd,meta=CMCTrumpet(args.mass,args.initial_quadrature_order).data(self.r)
            alpha=np.ones(args.n);alpha[:-1]=self.omega_from_grid()[:-1]*np.array([z['physical_lapse'] for z in meta[:-1]])
            self.alpha_profile=(alpha,self.r+rd[:,6],1+rdd[:,6],ref[:,6],rd[:,6],rdd[:,6])
        if getattr(args,'lapse_variable','deviation')=='log-reference':
            self.alpha_profile=((1+self.r*self.r)/2,self.r,np.ones(args.n),np.zeros(args.n),np.zeros(args.n),np.zeros(args.n))
        self.rhs_seconds=0.;self.elliptic_differential_residual=0.
        self.omega=(1-self.r*self.r)/2
        self.scri_quotient_weights=weights(self.r[-args.width:-1],1.)
        self.mass_jet_weights=[weights(self.r[-args.width:],1.,k) for k in [1,2,3]]

    def omega_from_grid(self):return (1-self.r*self.r)/2

    def from_physical(self,q,total_chi=None):
        result=q.copy()
        if self.metric_scale is not None:
            scale,base=self.metric_scale;inner=self.r<.5
            result[:,0]=np.log1p((q[:,0]-base)/scale)
            if total_chi is not None:result[inner,0]=np.log(total_chi[inner]/scale[inner])
        if self.alpha_profile is not None:
            alpha=self.alpha_profile[0];u=self.alpha_profile[3];result[:,6]=np.log1p((q[:,6]-u)/alpha)
        return result

    def to_physical(self,q):
        result=q.copy()
        with np.errstate(over='raise',invalid='raise'):
            if self.alpha_profile is not None:
                alpha=self.alpha_profile[0];u=self.alpha_profile[3];result[:,6]=u+alpha*np.expm1(q[:,6])
            if self.metric_scale is not None:
                scale,base=self.metric_scale;result[:,0]=base+scale*np.expm1(q[:,0])
        return result

    def parts(self,q):
        parity=PARITY.copy()
        if getattr(self.args,'origin_variables',False):parity[5]=1
        d,dd=self.op.derivatives(q,parity)
        total_alpha=None;total_chi=None;trumpet_chi=None;stored=q
        if self.metric_scale is not None:
            q=self.to_physical(stored)
            if self.trumpet_metric:
                A=(1+self.r*self.r)/2
                with np.errstate(over='raise',invalid='raise'):X=np.exp(stored[:,0])/A**2
                total_chi=self.r*self.r*X;ell=d[:,0]-2*self.r/A;ellp=dd[:,0]-2/A+2*self.r*self.r/A**2
                trumpet_chi=dict(X=X,ell=ell,ellp=ellp)
                d[:,0]=2*self.r*X+total_chi*ell
                dd[:,0]=2*X+4*self.r*X*ell+total_chi*(ellp+ell*ell)
            else:
                with np.errstate(over='raise',invalid='raise'):total_chi=np.exp(stored[:,0])
                dd[:,0]=total_chi*(dd[:,0]+d[:,0]**2);d[:,0]*=total_chi
        if self.alpha_profile is not None:
            alpha,ap,app,u,base_up,base_upp=self.alpha_profile
            with np.errstate(over='raise',invalid='raise'):factor=np.exp(stored[:,6])
            q=self.to_physical(stored);total_alpha=alpha*factor
            small=np.expm1(stored[:,6])
            dd[:,6]=base_upp+app*small+factor*(2*ap*d[:,6]+alpha*(dd[:,6]+d[:,6]**2))
            d[:,6]=base_up+ap*small+alpha*factor*d[:,6]
        if getattr(self.args,'origin_variables',False):
            # Invertible regular-center variables: (g-1)/r^2, A_rr/r^2,
            # Lambda/r. Curvature/connection remain independently evolved.
            stored=q;q=q.copy();dp=d.copy();ddp=dd.copy()
            for field,power in [(1,2),(2,2),(5,1)]:
                q[:,field]=self.r**power*stored[:,field]
                dp[:,field]=power*self.r**(power-1)*stored[:,field]+self.r**power*d[:,field]
                ddp[:,field]=(power*(power-1)*self.r**(power-2)*stored[:,field]
                    +2*power*self.r**(power-1)*d[:,field]+self.r**power*dd[:,field])
            d,dd=dp,ddp
        self.geometry_state=q;self.geometry_first=d;self.geometry_second=dd
        a=1+q[:,1];chi=1+q[:,0];alpha=(1+self.r*self.r)/2+q[:,6]
        if total_chi is not None:chi=total_chi
        if total_alpha is not None:alpha=total_alpha
        self.total_alpha=alpha
        self.total_chi=chi
        if min(np.min(a),np.min(chi),np.min(alpha))<=0:
            raise FloatingPointError('Nonpositive metric or lapse')
        start=time.perf_counter()
        if getattr(self.args,'backend','native')=='spherical-g1':
            import g1_rhs
            timing={}
            R,S,diag,beta,db,ddb=g1_rhs.parts(self.op,q,d,dd,self.args.kappa1,self.args.kappa2,
                self.args.theta=='appendix-c',self.args.slicing,0. if getattr(self.args,'lapse_restoring_form','symmetric')=='multiplicative' else self.restoring_strength,timing,
                puncture_cancellation=getattr(self.args,'puncture',False),total_alpha=total_alpha,total_chi=total_chi,
                trumpet_chi=trumpet_chi)
            self.solve_seconds+=timing['elliptic_seconds']
            self.shift_deviation=timing['beta_deviation'];self.origin_shift_slope=timing['origin_shift_slope']
            self.mass_numerator=timing['mass_numerator']
        else:
            beta,db,ddb,Q=solve(self.op,a,d[:,1],dd[:,1],alpha,self.r+d[:,6],
                q[:,2],d[:,2],-alpha[-1]*np.sqrt(chi[-1]/a[-1]),eta=self.args.eta)
            self.solve_seconds+=time.perf_counter()-start
            R,S,diag=self.kernel.parts(self.r,q,d,dd,beta,db,ddb,self.args.kappa1,
                self.args.kappa2,self.args.theta=='appendix-c',self.args.slicing,lapse_restoring=self.args.lapse_restoring)
        self.calls+=1
        if getattr(self.args,'lapse_restoring_form','symmetric')=='multiplicative':
            # -2xi alpha*(alpha-A) replaces -xi(alpha+A)*(alpha-A).
            # Their difference is quadratic and preserves alpha=0 at the puncture.
            if getattr(self.args,'backend','native')=='spherical-g1':S[:,6]-=2*self.restoring_strength*alpha*q[:,6]
            else:S[:,6]-=self.args.lapse_restoring*q[:,6]**2
        if getattr(self.args,'lapse_restoring_profile','constant')!='constant' and self.args.backend!='spherical-g1':
            factor=2*alpha if getattr(self.args,'lapse_restoring_form','symmetric')=='multiplicative' else alpha+(1+self.r*self.r)/2
            S[:,6]-=(self.restoring_strength-self.args.lapse_restoring)*factor*q[:,6]
        if getattr(self.args,'damping_form','constant')=='reference-lapse':
            D=-self.args.kappa1*q[:,6]
            S[:,3]+=D*(1-self.args.kappa2)*q[:,4]
            S[:,4]-=D*(2+self.args.kappa2)*q[:,4]
            S[:,5]-=D*q[:,5]
        if getattr(self.args,'slicing_family','linear')=='rational':
            S[:,6]+=rational_correction(self.r,alpha,q[:,6],q[:,3],self.args.slicing,self.args.slicing_scale)
        if getattr(self.args,'slicing_family','linear')=='blended-log':S[:,6]+=blended_correction(self.r,alpha,q[:,3],self.args.slicing)
        if getattr(self.args,'lapse_background_source','reference')=='collapsed-inner':
            S[:,6]+=collapsed_background_correction(self.r,q[:,6])
        S[:,6]+=self.lapse_source
        if getattr(self.args,'advection','centered')=='upwind':
            adv=self.op.advection_derivative(q,PARITY,beta)
            R[:,:7]+=beta[:,None]*(adv[:,:7]-d[:,:7])
        if getattr(self.args,'initial_data',None) in ('wormhole','trumpet'):
            F=lapse_coefficient(self.r[0],alpha[0],self.args.slicing,getattr(self.args,'slicing_family','linear'),getattr(self.args,'slicing_scale',.1))
            light=-beta[0]+alpha[0]*np.sqrt(chi[0]);gauge=-beta[0]+np.sqrt(F*chi[0])
            self.stage_metrics['max_inner_physical_characteristic']=max(self.stage_metrics.get('max_inner_physical_characteristic',-float('inf')),float(light))
            self.stage_metrics['max_inner_gauge_characteristic']=max(self.stage_metrics.get('max_inner_gauge_characteristic',-float('inf')),float(gauge))
            if light>0:raise FloatingPointError('Puncture inner boundary has an incoming physical characteristic')
            if gauge>0 and getattr(self.args,'puncture_boundary','outflow')=='outflow':
                raise FloatingPointError('Puncture inner boundary has an incoming gauge characteristic without boundary data')
        # Independent sampled-shift residual; do not use the analytic derivatives
        # returned by solve() for this diagnostic.
        if getattr(self.args,'backend','native')=='spherical-g1':
            from g1_rhs import radial_ratio
            fd,fdd=self.op.derivatives(self.shift_deviation,-1)
            vr=radial_ratio(self.shift_deviation,self.r,fd[0])
            term=radial_ratio(fd-vr,self.r,0.)
            er=4/3*(fdd+2*term)-2*((self.r+d[:,6])*q[:,2]+alpha*d[:,2])-6*alpha*radial_ratio(q[:,2],self.r,0.)
        else:
            fd, fdd=self.op.derivatives(beta,-1)
            er=residual(self.r,a,d[:,1],dd[:,1],alpha,self.r+d[:,6],q[:,2],d[:,2],beta,fd,fdd,self.args.eta)
        self.elliptic_differential_residual=float(np.max(np.abs(er)))
        # Exact nonlinear regularity traces for alpha(S)=1, Omega=(1-r^2)/2.
        self.regularity=np.array([chi[-1]-a[-1],0.,
            q[-1,2]-d[-1,1]/6+2*d[-1,0]/3,q[-1,3],q[-1,4],
            q[-1,5]-connection(self.r[-1],a[-1],d[-1,1]),0.,0.])
        return R,S,diag,beta

    def rhs(self,q,accepted=False):
        start=time.perf_counter();R,S,diag,beta=self.parts(q)
        if getattr(self.args,'origin_variables',False):
            R=R.copy();S=S.copy()
            for field,power in [(1,2),(2,2),(5,1)]:
                R[:,field]/=self.r**power;S[:,field]/=self.r**power
        result=R.copy();result[:-1]+=S[:-1]/self.omega[:-1,None]
        parity=PARITY.copy() if q.shape[1]==8 else np.r_[PARITY,1,1]
        if getattr(self.args,'origin_variables',False):parity[5]=1
        dS,_=self.op.derivatives(S,parity)
        if self.args.endpoint=='analytic-g1':
            import scri_limits
            derivative,_=scri_limits.derivatives(self.geometry_state[-1],self.geometry_first[-1],self.geometry_second[-1],
                self.shift_deviation[-1],self.args.kappa1,self.args.kappa2,self.args.theta=='appendix-c',self.args.slicing,self.args.lapse_restoring)
            if getattr(self.args,'lapse_restoring_form','symmetric')=='multiplicative':derivative[6]-=2*self.args.lapse_restoring*q[-1,6]*self.geometry_first[-1,6]
            if getattr(self.args,'damping_form','constant')=='reference-lapse':
                u=q[-1,6];up=self.geometry_first[-1,6];T=q[-1,4];Tp=self.geometry_first[-1,4];lam=q[-1,5];lp=self.geometry_first[-1,5]
                derivative[3]-=self.args.kappa1*(1-self.args.kappa2)*(up*T+u*Tp)
                derivative[4]+=self.args.kappa1*(2+self.args.kappa2)*(up*T+u*Tp)
                derivative[5]+=self.args.kappa1*(up*lam+u*lp)
            result[-1]=R[-1];result[-1,:8]-=derivative
        elif self.args.endpoint in ('raw-lhopital','compatible-trace'):
            result[-1]=R[-1]-dS[-1] # Omega'(1)=-1; full combined numerator.
        else:
            # Reconstruct the regular quotient p in S=Omega*p. This discretizes
            # the same limiting identity using interior combined numerators.
            # Keep the independently evaluated S(scri) for compatibility tests;
            # never reset a field or a constraint to force it to vanish.
            result[-1]=R[-1]+self.scri_quotient_weights @ (S[-self.args.width:-1]/self.omega[-self.args.width:-1,None])
        self.raw_endpoint=result[-1].copy()
        if self.args.endpoint=='compatible-trace':
            # Tangent evolution of analytically required regularity traces.
            # Initial data must satisfy them; no post-step field reset occurs.
            result[-1,0]=result[-1,1]  # chi(S)=g(S)
            result[-1,3]=0.  # p(S)=-3
            result[-1,4]=0.  # T(S)=0 in the compatible weighted phase
            result[-1,5]=-self.args.eta*(q[-1,5]-self.regularity[5])
            drate,_=self.op.derivatives(result,parity)
            result[-1,2]=drate[-1,1]/6-2*drate[-1,0]/3
        # Weak residual penalty; vanishes identically on the regular continuum
        # phase. It damps boundary incompatibility without resetting any field.
        result[-1]+=self.args.scri_sat/self.h*S[-1]
        result[-1]-=self.args.regularity_sat/self.h*self.regularity
        if getattr(self.args,'gauge_corner_sat',0.):
            import scri_limits
            correction,_=scri_limits.corner_correction(self.geometry_state[-1],self.geometry_first[-1],
                self.geometry_second[-1],R[-1,:8],self.shift_deviation[-1],self.h,self.args.gauge_corner_sat,
                1/self.op.d1[-1,-2],kappa1=self.args.kappa1,kappa2=self.args.kappa2,
                appendix_c=self.args.theta=='appendix-c',slicing=self.args.slicing,restoring=self.args.lapse_restoring,
                restoring_form=getattr(self.args,'lapse_restoring_form','symmetric'))
            result[-2,[0,3]]+=correction
        if self.alpha_profile is not None:result[:,6]/=self.total_alpha
        if self.metric_scale is not None:result[:,0]/=self.total_chi
        if self.args.origin_ko:
            # KO6 on regular spherical tensor coefficients, with a fixed
            # C5 taper to zero at r0. No scri/constraint values are reset.
            z=q.copy()
            if not getattr(self.args,'origin_variables',False):
                positive=self.r>0
                z[positive,2]/=self.r[positive]**2;z[positive,5]/=self.r[positive]
                if self.r[0]==0:z[0,2]=self.geometry_second[0,2]/2;z[0,5]=self.geometry_first[0,5]
            parity=parity.copy();parity[5]=1
            reflected=z[1:4][::-1] if self.r[0]==0 else z[:3][::-1]
            ext=np.concatenate([reflected*parity,z,np.zeros((3,q.shape[1]))])
            ko=np.zeros_like(q)
            for j,weight in enumerate([1,-6,15,-20,15,-6,1]):ko+=weight*ext[j:j+len(q)]
            if not getattr(self.args,'origin_variables',False):ko[:,2]*=self.r*self.r;ko[:,5]*=self.r
            ko[:,1]=0.;ko[:,7]=0.
            taper=np.maximum(1-(self.r/self.args.origin_ko_radius)**2,0.)**6
            result+=self.args.origin_ko*taper[:,None]*ko/(64*self.h)
        result[-1,6]=0. # Boundary clock on lapse only.
        result[:,7]=0. # Shift is recomputed by elliptic solve at every RK stage.
        if getattr(self.args,'puncture_boundary','outflow')=='gauge-robin':
            from gauge import robin_penalty
            alpha=diag[0,7];chi=diag[0,8]
            F=lapse_coefficient(self.r[0],alpha,self.args.slicing,getattr(self.args,'slicing_family','linear'),getattr(self.args,'slicing_scale',.1))
            penalty,residual,speed=robin_penalty(self.r[0],self.omega[0],alpha,chi,beta[0],
                self.r[0]+self.geometry_first[0,6],F,self.h*self.op.norm[0],self.args.puncture_lapse_power)
            result[0,3]+=penalty;result[0,2]+=2*penalty/(3*self.omega[0])
            self.stage_metrics['max_inner_gauge_incoming_speed']=max(self.stage_metrics.get('max_inner_gauge_incoming_speed',0.),speed)
            self.stage_metrics['max_inner_gauge_Robin_residual']=max(self.stage_metrics.get('max_inner_gauge_Robin_residual',0.),abs(residual))
            active=abs(residual)*speed/np.sqrt(F*chi)
            self.stage_metrics['max_active_inner_gauge_Robin_residual']=max(self.stage_metrics.get('max_active_inner_gauge_Robin_residual',0.),float(active))
        if getattr(self.args,'puncture_ko',0.):
            from numerics import puncture_dissipation
            diss=puncture_dissipation(q,self.r,self.h,self.args.puncture_ko,self.args.puncture_ko_radius,include_first=True)
            diss[:,1]=0.;diss[:,7]=0.;result+=diss/self.op.norm[:,None]
        self.stage_calls+=1
        maxima=dict(max_scri_pole=float(max(abs(S[-1,:6]))),max_lapse_pole=float(abs(S[-1,6])),
            max_regularity_trace=float(max(abs(self.regularity))),max_scri_null=float(abs(diag[-1,6])),max_scri_clock=float(abs(self.geometry_state[-1,6])))
        if getattr(self.args,'backend','native')=='spherical-g1':
            from bondi import nonaffinity
            maxima['max_Bondi_nonaffinity']=float(abs(nonaffinity(diag[-1,8],diag[-1,7],
                self.geometry_first[-1,0],1+self.geometry_first[-1,6],beta[-1]+1.5*diag[-1,7]*self.geometry_state[-1,2],result[-1,0]*(self.total_chi[-1] if self.metric_scale is not None else 1))))
        for i,name in enumerate(['H','M','Theta','Z']):maxima['max_'+name+'_rms']=float(np.sqrt(np.sum(self.r**2*diag[:,i]**2)/np.sum(self.r**2)))
        if hasattr(self.op,'shift_ode_solver'):maxima['max_elliptic_discrete_residual']=self.op.shift_ode_solver.last_residual
        if accepted:
            for i,name in enumerate(['H','M','Theta','Z']):maxima['max_accepted_'+name+'_rms']=maxima['max_'+name+'_rms']
            for i,name in enumerate(['H','M']):
                maxima['max_accepted_'+name+'_bulk']=float(max(abs(diag[self.r<.8,i])))
                maxima['max_accepted_'+name+'_endpoint']=float(max(abs(diag[self.r>.9,i])))
        for key,value in maxima.items():self.stage_metrics[key]=max(self.stage_metrics.get(key,0.),value)
        for key,value in [('min_chi',float(min(diag[:,8]))),('min_alpha',float(min(diag[:,7])))]:self.stage_metrics[key]=min(self.stage_metrics.get(key,float('inf')),value)
        if not np.all(np.isfinite(result)):raise FloatingPointError('Nonfinite RHS')
        self.rhs_seconds+=time.perf_counter()-start
        return result

    def report(self,q,t):
        R,S,c,b=self.parts(q);shape=self.r*self.r
        norm=lambda x:float(np.sqrt(np.sum(shape*x*x)/np.sum(shape)))
        core=self.r<.8;end=self.r>.9
        names=['H','M','Theta','Z']
        physical_points=self.r>0 if getattr(self.args,'puncture_chart',False) else np.ones(len(self.r),dtype=bool)
        row={'t':float(t),'max_field':float(np.max(np.abs(self.geometry_state[:,:7]))),
            'min_alpha':float(np.min(c[physical_points,7])),'min_chi':float(np.min(c[physical_points,8])),
            'min_g':float(np.min(c[:,9])),'scri_null':float(c[-1,6]),
            'scri_clock':float(self.geometry_state[-1,6]),'scri_pole':float(np.max(abs(S[-1,:6]))),
            'scri_lapse_pole':float(abs(S[-1,6])),
            'regularity_trace_max':float(max(abs(self.regularity))),
            'elliptic_sampled_residual':self.elliptic_differential_residual,
            'elliptic_calls':self.calls,'elliptic_seconds':self.solve_seconds,
            'det_max':float(max(abs(c[:,4]))),'tracefree_max':float(max(abs(c[:,5]))),
            'light_speed_min':float(min(-b-np.sqrt(c[:,8]/c[:,9])*c[:,7])),
            'light_speed_max':float(max(-b+np.sqrt(c[:,8]/c[:,9])*c[:,7]))}
        for i,name in enumerate(names):
            row[name+'_rms']=norm(c[:,i]);row[name+'_bulk_max']=float(max(abs(c[core,i])))
            row[name+'_endpoint_max']=float(max(abs(c[end,i])))
            # The linear theorem weights are diagnostic, not imposed constraints.
            power=4 if name in ('H','Theta') else 3
            row[name+'_weighted_interior_max']=float(max(abs(c[:-1,i]/self.omega[:-1]**power)))
        if hasattr(self.op,'shift_ode_solver'):
            row['elliptic_discrete_residual']=self.op.shift_ode_solver.last_residual
            row['elliptic_backward_error']=self.op.shift_ode_solver.last_backward_error
        if self.stage_metrics:row['all_RK_stages']=dict(self.stage_metrics)
        for location in [.25,.5,.75,.9,.98]:
            row['mass_r'+str(location)]=float(np.interp(location,self.r[:-1],c[:-1,10]))
        if getattr(self.args,'puncture',False):
            first=1 if getattr(self.args,'puncture_chart',False) else 0
            rr=self.r[first:-1];radius=rr/(self.omega[first:-1]*np.sqrt(c[first:-1,8]));density=radius/rr
            alpha=c[:,7];F=lapse_coefficient(self.r,alpha,self.args.slicing,getattr(self.args,'slicing_family','linear'),getattr(self.args,'slicing_scale',.1))
            row.update(inner_areal_radius=float(radius[0]),inner_proper_distance_density=float(density[0]),
                inner_light_speed_max=float(-b[first]+alpha[first]*np.sqrt(c[first,8])),
                inner_gauge_speed_max=float(-b[first]+np.sqrt(F[first]*c[first,8])),
                inner_radius_log_slope=float(np.polyfit(np.log(rr[:8]),np.log(radius[:8]),1)[0]),
                inner_proper_distance_log_slope=float(np.polyfit(np.log(rr[:8]),np.log(density[:8]),1)[0]),
                inner_physical_lapse_log_slope=float(np.polyfit(np.log(rr[:8]),np.log(alpha[first:first+8]/self.omega[first:first+8]),1)[0]))
        if getattr(self.args,'backend','native')=='spherical-g1':
            from bondi import nonaffinity
            # Compute the endpoint metric rate of this numerical boundary
            # method. Affine coordinate time must be measured, not assumed.
            if self.args.endpoint=='analytic-g1':
                import scri_limits
                ds,_=scri_limits.derivatives(self.geometry_state[-1],self.geometry_first[-1],self.geometry_second[-1],
                    self.shift_deviation[-1],self.args.kappa1,self.args.kappa2,self.args.theta=='appendix-c',self.args.slicing,self.args.lapse_restoring)
                ct=R[-1,0]-ds[0]
            elif self.args.endpoint=='quotient-extrapolation':
                ct=R[-1,0]+self.scri_quotient_weights@(S[-self.args.width:-1,0]/self.omega[-self.args.width:-1])
            else:ct=R[-1,0]-self.op.endpoint_first(S[:,0])
            if self.args.endpoint=='compatible-trace':ct=0.
            ct+=self.args.scri_sat/self.h*S[-1,0]-self.args.regularity_sat/self.h*self.regularity[0]
            if getattr(self.args,'puncture_chart',False) and self.args.advection=='upwind':
                adv=self.op.advection_derivative(q,np.ones(8),b)
                ct+=c[-1,8]*b[-1]*(adv[-1,0]-self.chart_first[-1,0])
            db=b[-1]+1.5*c[-1,7]*self.geometry_state[-1,2]
            row['Bondi_frame_nonaffinity']=float(nonaffinity(c[-1,8],c[-1,7],self.geometry_first[-1,0],
                1+self.geometry_first[-1,6],db,ct))
            row['scri_chi_time_rate']=float(ct)
            # Finite-mass expansion N=Omega^3 F gives m(S)=-N'''(S)/(12chi^1.5).
            # Expose lower jets: this limit is inadmissible if they do not vanish.
            jet=[float(w@self.mass_numerator[-self.args.width:]) for w in self.mass_jet_weights]
            row.update(mass_numerator_scri=float(self.mass_numerator[-1]),
                mass_numerator_d1_scri=jet[0],mass_numerator_d2_scri=jet[1],
                mass_scri_third_jet=-jet[2]/(12*c[-1,8]**1.5))
        row.update(apparent_horizons(self.r,c[:,8],c[:,11]))
        return row,b,c


class CoupledEvolution(Evolution):
    def parts(self,q):
        import scalar_rhs
        geom=q[:,:8];fields=q[:,8:]
        R,S,diag,beta=super().parts(geom)
        geom=self.geometry_state;d=self.geometry_first;dd=self.geometry_second
        df,ddf=self.op.derivatives(fields,np.ones(2))
        phi,phir,normal=scalar_rhs.matter(self.op,geom,d,fields,df,beta,total_chi=self.total_chi)
        scalar_rhs.add_sources(self.r,geom,R,diag,phir,normal,total_chi=self.total_chi,total_alpha=self.total_alpha)
        # Use the same stage solve, including its analytically small deviation.
        v=self.shift_deviation
        wave,info=scalar_rhs.rhs(self.op,geom,d,fields,df,ddf,beta,v,
            self.args.kappa1,self.args.kappa2,self.args.theta=='appendix-c',geometry_second=dd,origin_beta_derivative=self.origin_shift_slope,
            analytic_endpoint=self.args.endpoint=='analytic-g1',damping_form=getattr(self.args,'damping_form','constant'),total_chi=self.total_chi,total_alpha=self.total_alpha)
        self.scalar_info=info;self.scalar_phi=phi;self.scalar_phir=phir;self.scalar_normal=normal
        self.scalar_scri_rate=wave[-1,0]
        if getattr(self.args,'advection','centered')=='upwind':
            wave+=beta[:,None]*(self.op.advection_derivative(fields,np.ones(2),beta)-df)
            self.scalar_scri_rate=wave[-1,0]
        self.regularity=np.r_[self.regularity,0.,0.]
        return np.c_[R,wave],np.c_[S,np.zeros_like(wave)],diag,beta

    def report(self,q,t):
        row,b,c=super().report(q,t)
        B=self.scalar_info['potential_numerator'];dB,_=self.op.derivatives(B,1)
        row.update(scalar_max=float(max(abs(self.scalar_phi))),
            scalar_rescaled_scri=float(np.sqrt(self.total_chi[-1])*q[-1,8]),
            radius_weighted_scalar_scri=float(q[-1,8]),
            coordinate_time_scalar_flux_scri=float(4*np.pi*self.scalar_scri_rate**2),
            scalar_potential_pole0=float(B[-1]),scalar_potential_pole1=float(dB[-1]))
        return row,b,c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin-grid',action='store_true')
    p.add_argument('--fast-library',help='Optional audited compiled g=1 RK4 engine')
    p.add_argument('--gauge-corner-sat',type=float,default=0.)
    p.add_argument('--advection',choices=['centered','upwind'],default='centered')
    p.add_argument('--puncture',action='store_true')
    p.add_argument('--puncture-chart',action='store_true')
    p.add_argument('--puncture-ko',type=float,default=0.)
    p.add_argument('--puncture-ko-radius',type=float,default=.8)
    p.add_argument('--initial-data',choices=['minkowski','wormhole','trumpet'],default='minkowski')
    p.add_argument('--mass',type=float,default=.1)
    p.add_argument('--initial-quadrature-order',type=int,choices=[64,96,128,192],default=96)
    p.add_argument('--perturbation',choices=['none','constraint','noise','noise-balanced'],default='none')
    p.add_argument('--perturbation-amplitude',type=float,default=1e-8)
    p.add_argument('--seed',type=int,default=7419)
    p.add_argument('--origin-variables',action='store_true')
    p.add_argument('--scalar',action='store_true')
    p.add_argument('--scalar-amplitude',type=float,default=.01)
    p.add_argument('--initial-tolerance',type=float,default=1e-10)
    p.add_argument('--initial-precision',choices=['float','decimal'],default='float')
    p.add_argument('--initial-lapse',choices=['reference','stationary-chi'],default='reference')
    p.add_argument('--origin-ko',type=float,default=0.)
    p.add_argument('--origin-ko-radius',type=float,default=.4)
    p.add_argument('--backend',choices=['native','spherical-g1'],default='native')
    p.add_argument('--quadrature-width',type=int,choices=[5,7,9,11],default=7)
    p.add_argument('--elliptic-quadrature',choices=['prefix','inward'],default='prefix')
    p.add_argument('--elliptic-method',choices=['quadrature','discrete-ode'],default='quadrature')
    p.add_argument('--elliptic-refinements',type=int,choices=[0,1,2,4],default=2)
    p.add_argument('--elliptic-bias',choices=['centered','right'],default='centered')
    p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=64);p.add_argument('--t',type=float,default=1.)
    p.add_argument('--cfl',type=float,default=.05);p.add_argument('--amplitude',type=float,default=0.)
    p.add_argument('--pulse',choices=['compact','published-gaussian'],default='compact')
    p.add_argument('--center',type=float,default=.25);p.add_argument('--pulse-width',type=float,default=.1)
    p.add_argument('--kappa1',type=float,default=4.);p.add_argument('--kappa2',type=float,default=-.99)
    p.add_argument('--damping-form',choices=['constant','reference-lapse'],default='constant')
    p.add_argument('--lapse-restoring',type=float,default=0.)
    p.add_argument('--lapse-restoring-form',choices=['symmetric','multiplicative'],default='symmetric')
    p.add_argument('--lapse-restoring-profile',choices=['constant','physical-log'],default='constant')
    p.add_argument('--eta',type=float,default=0.);p.add_argument('--slicing',type=float,default=1.)
    p.add_argument('--slicing-family',choices=['linear','rational','blended-log'],default='linear')
    p.add_argument('--slicing-scale',type=float,default=.1)
    p.add_argument('--lapse-background-source',choices=['reference','collapsed-inner'],default='reference')
    p.add_argument('--lapse-reference',choices=['minkowski','CMC-trumpet'],default='minkowski')
    p.add_argument('--lapse-variable',choices=['deviation','log-reference','log-trumpet'],default='deviation')
    p.add_argument('--metric-variable',choices=['deviation','log','log-trumpet'],default='deviation')
    p.add_argument('--inner-radius',type=float,default=0.)
    p.add_argument('--theta',choices=['appendix-c','repository'],default='appendix-c')
    p.add_argument('--width',type=int,choices=[5,7,9],default=7)
    p.add_argument('--radial-operator',choices=['fd','sbp63'],default='fd')
    p.add_argument('--puncture-boundary',choices=['outflow','gauge-robin'],default='outflow')
    p.add_argument('--puncture-lapse-power',type=float,default=2.)
    p.add_argument('--endpoint',choices=['raw-lhopital','analytic-g1','quotient-extrapolation','compatible-trace'],default='raw-lhopital')
    p.add_argument('--regularity-sat',type=float,default=0.)
    p.add_argument('--scri-sat',type=float,default=0.)
    p.add_argument('--report-dt',type=float,default=.1);p.add_argument('--resume',type=Path)
    args=p.parse_args()
    numeric=[args.cfl,args.t,args.amplitude,args.center,args.pulse_width,args.kappa1,args.kappa2,args.eta,args.slicing,args.report_dt,args.origin_ko,args.origin_ko_radius,args.lapse_restoring,args.scalar_amplitude,args.initial_tolerance,args.perturbation_amplitude,args.gauge_corner_sat,args.mass,args.puncture_ko,args.puncture_ko_radius,args.slicing_scale,args.puncture_lapse_power,args.inner_radius]
    if not all(np.isfinite(x) for x in numeric):raise ValueError('Nonfinite input')
    if (args.n<16 or args.cfl<=0 or args.t<0 or args.pulse_width<=0 or not 0<=args.center<1
        or args.report_dt<=0 or args.origin_ko<0 or not 0<args.origin_ko_radius<=.75
        or args.kappa1<0 or args.eta<0 or args.slicing<0 or args.lapse_restoring<0
        or args.initial_tolerance<=0 or args.gauge_corner_sat<0 or args.mass<=0 or args.puncture_ko<0
        or not 0<args.puncture_ko_radius<.95 or args.slicing_scale<=0 or args.puncture_lapse_power<0 or not 0<=args.inner_radius<1):raise ValueError('Invalid input')
    if args.inner_radius and (not args.puncture or args.initial_data!='trumpet' or args.puncture_chart):
        raise ValueError('Fixed-radius excision is a separate already-trumpet control; it does not replace wormhole evolution')
    if args.elliptic_method=='discrete-ode' and (args.backend!='spherical-g1' or args.radial_operator!='fd' or args.puncture_chart):
        raise ValueError('Discrete shift ODE prototype requires the g=1 FD regular-center or fixed-cutoff path')
    if args.slicing_family=='blended-log' and args.slicing==0:raise ValueError('Blended-log slicing requires a positive interior strength')
    if args.lapse_reference=='CMC-trumpet' and (args.initial_data!='trumpet' or args.lapse_background_source!='reference'):
        raise ValueError('The CMC lapse-reference control requires already-trumpet data and the reference background source')
    if args.lapse_variable=='log-trumpet' and (args.initial_data!='trumpet' or args.fast_library or args.advection!='centered'):
        raise ValueError('Log-trumpet lapse variables currently require Python already-trumpet evolution and centered transport')
    if args.metric_variable=='log-trumpet' and (args.initial_data!='trumpet' or args.fast_library or args.advection!='centered'):
        raise ValueError('Log-trumpet metric variables currently require Python already-trumpet evolution and centered transport')
    if (args.metric_variable!='deviation' or args.lapse_variable!='deviation') and (args.backend!='spherical-g1' or args.advection!='centered' or args.puncture_chart):
        raise ValueError('Positive-variable prototypes require g=1 centered evolution and their own representation')
    if (args.metric_variable!='deviation' or args.lapse_variable!='deviation') and args.origin_variables:
        raise ValueError('Combined positive and scaled-origin charts have not been implemented/audited')
    if args.radial_operator=='sbp63' and (not args.puncture or args.advection!='centered' or args.fast_library):
        raise ValueError('The SBP prototype requires a Python puncture evolution and centered transport')
    if args.puncture_boundary=='gauge-robin' and (not args.puncture or args.puncture_chart or args.radial_operator!='sbp63'):
        raise ValueError('Incoming-gauge Robin SAT requires the finite-cutoff SBP puncture path')
    if args.puncture_ko and not (args.puncture_chart or (args.puncture and args.radial_operator=='sbp63')):
        raise ValueError('Puncture dissipation requires an AF chart or finite-cutoff SBP path')
    if args.puncture and (args.backend!='spherical-g1' or args.origin_grid or args.origin_variables or args.origin_ko):
        raise ValueError('Puncture uses unscaled g=1, excluded inner endpoint and no regular-center KO filter')
    if args.initial_data in ('wormhole','trumpet') and (not args.puncture or args.scalar):raise ValueError('Black-hole initialization currently requires the vacuum puncture path')
    if args.puncture and args.initial_data not in ('wormhole','trumpet'):raise ValueError('The puncture path requires black-hole initial data')
    if args.puncture_chart and (not args.puncture or args.initial_data!='wormhole' or args.fast_library or args.lapse_restoring_form!='multiplicative'
        or args.perturbation!='none' or args.amplitude or args.gauge_corner_sat or args.endpoint not in ('raw-lhopital','analytic-g1')):
        raise ValueError('AF coefficient prototype requires vacuum wormhole, Python RK, multiplicative restoring and an unperturbed initial lapse')
    if args.origin_grid and (args.backend!='spherical-g1' or args.origin_variables):raise ValueError('Actual-origin grid uses the analytic g=1 limits with unscaled fields')
    if args.endpoint=='analytic-g1' and (args.backend!='spherical-g1' or args.origin_variables):raise ValueError('Analytic endpoint requires unscaled g=1')
    if args.gauge_corner_sat and (args.backend!='spherical-g1' or args.origin_variables):raise ValueError('Gauge corner penalty requires unscaled g=1')
    if args.fast_library and (args.backend!='spherical-g1' or args.origin_variables or args.endpoint not in ('raw-lhopital','analytic-g1')):
        raise ValueError('Compiled engine requires unscaled g=1 and a supported endpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    # Bind all actual header dependencies, adapter and Python implementation.
    module_names=['evolve.py','numerics.py','elliptic.py','kernel.py','gauge.py','bondi.py','horizon.py','time_lattice.py']
    if args.radial_operator=='sbp63':module_names+=['sbp.py']
    if args.elliptic_method=='discrete-ode':module_names+=['shift_ode.py']
    if args.backend=='spherical-g1':module_names+=['g1_rhs.py']
    if args.scalar:module_names+=['scalar_rhs.py','scalar_initial_data.py']
    if args.fast_library:module_names+=['fast.py','fast_integrator.cpp']
    if args.endpoint=='analytic-g1' or args.gauge_corner_sat:module_names+=['scri_limits.py']
    if args.initial_data=='wormhole':module_names+=['wormhole_isotropic.py','wormhole_data.py']
    if args.initial_data=='trumpet':module_names+=['trumpet_data.py','wormhole_isotropic.py','wormhole_data.py']
    if args.puncture_chart:module_names+=['puncture_chart.py','scri_limits.py']
    source=[*sorted((root/'src/z4c/hyperboloidal').glob('*.hpp')),
            *[Path(__file__).with_name(name) for name in module_names],
            Path(__file__).with_name('spherical_api.cpp')]
    binding={'source_sha256':{str(f.relative_to(root)):sha(f) for f in source},
             'executable_sha256':sha(args.library)}
    if args.fast_library:binding['fast_executable_sha256']=sha(args.fast_library)
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'command':sys.argv,'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'binding':binding,'checkpoint_parent':None,'status':'running','method':'actual-scri free-Z4c prototype; inner endpoint specified in arguments',
        'resource_authorization':'2026-10-25 user: given the code will be run on 1 spatial dimension, just run the tests locally'}
    if args.scalar and args.backend!='spherical-g1':raise ValueError('Coupled scalar currently requires spherical-g1 backend')
    if args.puncture_chart:
        from puncture_chart import PunctureChart
        ev=PunctureChart(args);manifest['representation']='puncture-log-AF-v1'
    else:ev=CoupledEvolution(args) if args.scalar else Evolution(args)
    if args.lapse_variable=='log-trumpet':manifest['representation']='log-trumpet-lapse-v1; geometric/gauge reference selected independently'
    if args.metric_variable=='log-trumpet':manifest['representation']='log-trumpet-metric-v1; geometric/gauge reference selected independently'
    if args.metric_variable=='log' or args.lapse_variable=='log-reference':manifest['representation']='positive-reference-variables-v1; free Z4c fields retained'
    if args.inner_radius:manifest['inner_boundary_kind']='fixed excision control; stage characteristic guards required; no wormhole-to-trumpet claim'
    # Freeze before initialization too: a failed Newton solve is an experiment.
    snapshot=args.output/'frozen-source';snapshot.mkdir()
    for f in source:
        destination=snapshot/f.relative_to(root);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,destination)
        if sha(destination)!=binding['source_sha256'][str(f.relative_to(root))]:raise RuntimeError('Source changed during packaging')
    (snapshot/'runtime').mkdir();shutil.copy2(args.library,snapshot/'runtime'/Path(args.library).name)
    if args.fast_library:shutil.copy2(args.fast_library,snapshot/'runtime'/Path(args.fast_library).name)
    manifest['status']='initializing';(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    q=np.zeros((args.n,10 if args.scalar else 8));t=0.;step=0
    # Smooth compact pulse in r^2, exact regular parity, support strictly inside scri.
    x=(ev.r**2-args.center**2)/(2*args.center*args.pulse_width+args.pulse_width**2)
    mask=abs(x)<1
    q[mask,6]=args.amplitude*np.exp(1-1/(1-x[mask]**2))
    if args.pulse=='published-gaussian':
        q[:,6]=args.amplitude*np.exp(-(ev.r**2-args.center**2)**2/(4*args.pulse_width**4))
    if q[-1,6]!=0.:raise ValueError('Initial lapse pulse must vanish at scri for clock-compatible data')
    if args.initial_data=='wormhole' and not args.resume:
        from wormhole_isotropic import IsotropicWormhole
        initial=IsotropicWormhole(args.mass,args.initial_quadrature_order)
        if args.puncture_chart:q,iq,id1,id2,geometry=ev.initial_wormhole(initial)
        else:iq,id1,id2,geometry=initial.data(ev.r);q[:,:8]+=iq
        manifest['wormhole_initial_data']=dict(mass=args.mass,r0=initial.r0,isotropic_scale=initial.k,
            height='H=n(R)=s(R)R; s=0 below4M, s=1 above8M; h_R=H/[F sqrt(F+H^2)]',
            inner_lapse='physical psi_iso^-2, psi_iso=1+M/(2rho)',outer_lapse='physical sqrt(F+R^2)',
            reference='Minkowski; no Schwarzschild RHS subtraction',inner_boundary='evolved AF coefficients' if args.puncture_chart else 'one-sided outflow, no regular-center parity')
        np.savez(args.output/'wormhole-initial-data.npz',r=ev.r,q=iq if args.puncture_chart else q,q_stored=q,d=id1,dd=id2,
            chi_total=np.array([g['chi_total'] for g in geometry]),R_areal=np.array([g['R'] for g in geometry]))
    if args.initial_data=='trumpet' and not args.resume:
        from trumpet_data import CMCTrumpet
        initial=CMCTrumpet(args.mass,args.initial_quadrature_order)
        iq,id1,id2,geometry=initial.data(ev.r);q+=iq
        manifest['trumpet_initial_data']=dict(mass=args.mass,R0=initial.R0,C=initial.C,gamma=initial.gamma,
            K_CMC=-3.,H='R-C/R^2',lapse='stationary physical sqrt(1-2M/R+H^2)',
            reference='Minkowski; no Schwarzschild RHS subtraction',classification='Already-trumpet control; not wormhole-to-trumpet evidence')
        np.savez(args.output/'trumpet-initial-data.npz',r=ev.r,q=q,d=id1,dd=id2,
            chi_total=np.array([g['chi_total'] for g in geometry]),R_areal=np.array([g['R'] for g in geometry]))
    if args.scalar and not args.resume:
        from scalar_initial_data import solve_data
        initial_grid=ev.r if args.origin_grid else np.r_[0.,ev.r]
        initial=solve_data(len(initial_grid)-1,args.scalar_amplitude,args.initial_tolerance,grid=initial_grid,precision=args.initial_precision)
        use=slice(None) if args.origin_grid else slice(1,None)
        manifest['scalar_initial_data']={'amplitude':args.scalar_amplitude,'regular_residual':float(initial['residual']),
            'rounded_double_residual':initial['rounded_double_residual'],'arithmetic':args.initial_precision,
            'Newton_history':initial['iterations'],'Bondi_mass':float(8*initial['w'][-1]),'normal_physical_scalar':'n_bar(phi)=0'}
        if not initial['converged']:
            failure={'type':'InitialDataConvergenceFailure','message':'Requested Newton tolerance not achieved','time':0.}
            manifest.update(status='initial-data-failed',failure=failure)
            (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            np.savez(args.output/'failed-initial-data.npz',r=initial_grid,w=initial['w'],phi=initial['phi'])
            print(json.dumps(failure));return 1
        O=ev.omega;w=initial['w'][use];psi=initial['psi'][use]
        q[:,0]=np.expm1(-4*np.log1p(O**3*w))
        phi=initial['phi'][use]
        q[:-1,8]=phi[:-1]/(O[:-1]*np.sqrt(1+q[:-1,0]));q[-1,8]=0.
        q[:-1,9]=q[:-1,8]/O[:-1];q[-1,9]=0.
        if args.initial_lapse=='stationary-chi':
            from scalar_initial_data import stationary_chi_lapse
            q[:,6]+=stationary_chi_lapse(initial)[use]
    if args.perturbation!='none' and not args.resume:
        # Compact support strictly inside scri; independent free constraint
        # errors obey center tensor parity. No differential constraint solve.
        rng=np.random.default_rng(args.seed);x=ev.r/.7;envelope=np.zeros_like(x)
        inside=x<1;envelope[inside]=np.exp(1-1/(1-x[inside]**2))
        for field,power in [(0,0),(2,2),(3,0),(4,0),(5,1)]:
            if args.perturbation in ('noise','noise-balanced'):profile=rng.uniform(-1,1,args.n)
            else:profile=sum(rng.normal()*ev.r**(2*k) for k in range(4))
            scale=ev.h**(2 if field==0 else 1) if args.perturbation=='noise-balanced' else 1.
            q[:,field]+=args.perturbation_amplitude*scale*ev.r**power*envelope*profile
    if (args.lapse_variable!='deviation' or args.metric_variable!='deviation') and not args.resume:
        q=ev.from_physical(q,total_chi=np.array([g['chi_total'] for g in geometry]) if args.metric_variable=='log-trumpet' else None)
    if args.resume:
        with np.load(args.resume,allow_pickle=False) as z:
            old=json.loads(str(z['metadata']))
            if old['binding']!=binding:raise ValueError('Checkpoint source/library binding mismatch')
            for key in ('n','cfl','kappa1','kappa2','damping_form','eta','slicing','slicing_family','slicing_scale','lapse_background_source','lapse_reference','lapse_variable','metric_variable','inner_radius','theta','width','radial_operator','puncture_boundary','puncture_lapse_power','endpoint','scri_sat','regularity_sat','lapse_restoring','lapse_restoring_form','lapse_restoring_profile','backend','quadrature_width','elliptic_quadrature','elliptic_method','elliptic_refinements','elliptic_bias','origin_ko','origin_ko_radius','scalar','scalar_amplitude','initial_tolerance','initial_precision','initial_lapse','origin_variables','origin_grid','perturbation','perturbation_amplitude','seed','gauge_corner_sat','advection','puncture','puncture_chart','puncture_ko','puncture_ko_radius','initial_data','mass','initial_quadrature_order'):
                if old['arguments'][key]!=manifest['arguments'][key]:raise ValueError('Checkpoint input mismatch: '+key)
            q=z['q'].copy();t=float(z['t']);step=int(z['step'])
            for key in ['scalar_initial_data','wormhole_initial_data','trumpet_initial_data']:
                if key in old:manifest[key]=old[key]
        manifest['checkpoint_parent']={'path':str(args.resume.resolve()),'sha256':sha(args.resume)}
    radiated_energy=float(old.get('radiated_energy',0.)) if args.resume else 0.
    dt=args.cfl*ev.h
    clock=FixedStepClock(dt,t,step,args.t,old.get('time_lattice') if args.resume else None)
    t=clock.t;manifest['time_lattice']=clock.metadata()
    manifest['status']='running'
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.perf_counter();failure=None;history=[];report_start=t;report_index=0;next_report_step=step
    # Constant CFL until the final landing; resume reproduces aligned timesteps.
    compiled=None
    if args.fast_library:
        from fast import FastEngine
        compiled=FastEngine(ev,q.shape[1],args.fast_library)
    try:
        with (args.output/'history.jsonl').open('w') as log:
            while True:
                if step>=next_report_step or clock.complete:
                    row,b,c=ev.report(q,t);row['radiated_energy']=radiated_energy;row['step']=step
                    if compiled:
                        calls,seconds=compiled.stats();row['elliptic_calls']+=calls;row['elliptic_seconds']+=seconds
                        row['all_RK_stages']=compiled.diagnostics()
                    history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    np.savez(args.output/f'profile-{len(history):04d}.npz',r=ev.r,q=q,q_physical=np.c_[ev.geometry_state,q[:,8:]],beta=b,diagnostic=c,t=t,step=step,time_lattice=json.dumps(clock.metadata()))
                    report_index=max(report_index+1,int(np.floor((t-report_start)/args.report_dt))+1)
                    next_report_step=max(step+1,clock.report_step(report_start+report_index*args.report_dt))
                if clock.complete:break
                if compiled:
                    available=clock.full_steps_available
                    count=max(1,min(4096,available,next_report_step-step))
                    h=dt if available>=1 else clock.final_delta
                    done,radiated_energy,failed=compiled.advance(q,count,h,radiated_energy)
                    clock.advance(done,h);t=clock.t;step=clock.step
                    if failed:raise FloatingPointError(compiled.failure_message)
                    continue
                h=dt if clock.full_steps_available else clock.final_delta
                k1=ev.rhs(q,accepted=True);flux1=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                k2=ev.rhs(q+h*k1/2);flux2=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                k3=ev.rhs(q+h*k2/2);flux3=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                k4=ev.rhs(q+h*k3);flux4=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                q=q+h*(k1+2*k2+2*k3+k4)/6
                radiated_energy+=h*(flux1+2*flux2+2*flux3+flux4)/6
                clock.advance(1,h);t=clock.t;step=clock.step
    except Exception as e:
        failure={'type':type(e).__name__,'message':str(e),'time':t,'step':step}
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    finally:
        if compiled:
            calls,seconds=compiled.stats();ev.calls+=calls;ev.solve_seconds+=seconds
            manifest['all_RK_stages']=compiled.diagnostics();manifest['compiled_rk_seconds']=compiled.advance_seconds;compiled.close()
        if not compiled:manifest['all_RK_stages']=dict(ev.stage_metrics)
        manifest.update(status='failed' if failure else 'finished',time=t,steps=step,
            time_lattice=clock.metadata(),
            runtime_seconds=time.perf_counter()-start,rhs_seconds=ev.rhs_seconds,radiated_energy=radiated_energy,
            elliptic_seconds=ev.solve_seconds,elliptic_calls=ev.calls,failure=failure)
        extra={}
        if args.puncture_chart:
            try:
                physical,chi_total,alpha_total,_=ev.reconstruct(q)
                extra.update(q_physical=physical,chi_total=chi_total,alpha_total=alpha_total)
            except FloatingPointError:manifest['checkpoint_physical_reconstruction']='failed'
        elif args.lapse_variable!='deviation' or args.metric_variable!='deviation':
            try:
                physical=ev.to_physical(q)
                alpha_total=ev.alpha_profile[0]*np.exp(q[:,6]) if ev.alpha_profile is not None else (1+ev.r**2)/2+physical[:,6]
                chi_total=ev.metric_scale[0]*np.exp(q[:,0]) if ev.metric_scale is not None else 1+physical[:,0]
                extra.update(q_physical=physical,chi_total=chi_total,alpha_total=alpha_total)
            except FloatingPointError:manifest['checkpoint_physical_reconstruction']='failed'
        np.savez(args.output/'checkpoint.npz',q=q,r=ev.r,t=t,step=step,metadata=json.dumps(manifest),**extra)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('status','time','steps','runtime_seconds','failure')}))
    return 1 if failure else 0

if __name__=='__main__':sys.exit(main())
