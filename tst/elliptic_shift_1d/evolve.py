#!/usr/bin/env python3
"""Experimental actual-scri free-Z4c vacuum/scalar regular-center driver.

All failures are retained. Puncture evolution is not implemented here.
"""
import argparse,hashlib,json,os,platform,sys,time,shutil
from pathlib import Path
import numpy as np
from elliptic import solve,residual,connection
from numerics import RadialOperators,weights
from kernel import Kernel,PARITY


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evolution:
    def __init__(self,args):
        self.args=args;self.r=np.linspace(0,1,args.n) if getattr(args,'origin_grid',False) else (np.arange(args.n)+.5)/(args.n-.5)
        self.h=1/(args.n-1) if getattr(args,'origin_grid',False) else 1/(args.n-.5);self.op=RadialOperators(self.r,width=args.width,quadrature_width=args.quadrature_width)
        self.kernel=Kernel(args.library);self.calls=0;self.solve_seconds=0.
        self.rhs_seconds=0.;self.elliptic_differential_residual=0.
        self.omega=(1-self.r*self.r)/2
        self.scri_quotient_weights=weights(self.r[-args.width:-1],1.)
        self.mass_jet_weights=[weights(self.r[-args.width:],1.,k) for k in [1,2,3]]

    def parts(self,q):
        parity=PARITY.copy()
        if getattr(self.args,'origin_variables',False):parity[5]=1
        d,dd=self.op.derivatives(q,parity)
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
        if min(np.min(a),np.min(chi),np.min(alpha))<=0:
            raise FloatingPointError('Nonpositive metric or lapse')
        start=time.perf_counter()
        if getattr(self.args,'backend','native')=='spherical-g1':
            import g1_rhs
            timing={}
            R,S,diag,beta,db,ddb=g1_rhs.parts(self.op,q,d,dd,self.args.kappa1,self.args.kappa2,
                self.args.theta=='appendix-c',self.args.slicing,self.args.lapse_restoring,timing)
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

    def rhs(self,q):
        start=time.perf_counter();R,S,diag,beta=self.parts(q)
        if getattr(self.args,'origin_variables',False):
            R=R.copy();S=S.copy()
            for field,power in [(1,2),(2,2),(5,1)]:
                R[:,field]/=self.r**power;S[:,field]/=self.r**power
        result=R.copy();result[:-1]+=S[:-1]/self.omega[:-1,None]
        parity=PARITY.copy() if q.shape[1]==8 else np.r_[PARITY,1,1]
        if getattr(self.args,'origin_variables',False):parity[5]=1
        dS,_=self.op.derivatives(S,parity)
        if self.args.endpoint in ('raw-lhopital','compatible-trace'):
            result[-1]=R[-1]-dS[-1] # Omega'(1)=-1; full combined numerator.
        else:
            # Reconstruct the regular quotient p in S=Omega*p. This discretizes
            # the same limiting identity using interior combined numerators.
            # Keep the independently evaluated S(scri) for compatibility tests;
            # never reset a field or a constraint to force it to vanish.
            result[-1]=R[-1]+self.scri_quotient_weights @ (S[-self.args.width:-1]/self.omega[-self.args.width:-1,None])
        self.raw_endpoint=R[-1]-dS[-1]
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
        if not np.all(np.isfinite(result)):raise FloatingPointError('Nonfinite RHS')
        self.rhs_seconds+=time.perf_counter()-start
        return result

    def report(self,q,t):
        R,S,c,b=self.parts(q);shape=self.r*self.r
        norm=lambda x:float(np.sqrt(np.sum(shape*x*x)/np.sum(shape)))
        core=self.r<.8;end=self.r>.9
        names=['H','M','Theta','Z']
        row={'t':float(t),'max_field':float(np.max(np.abs(self.geometry_state[:,:7]))),
            'min_alpha':float(np.min(c[:,7])),'min_chi':float(np.min(c[:,8])),
            'min_g':float(np.min(c[:,9])),'scri_null':float(c[-1,6]),
            'scri_clock':float(q[-1,6]),'scri_pole':float(np.max(abs(S[-1,:6]))),
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
        for location in [.25,.5,.75,.9,.98]:
            row['mass_r'+str(location)]=float(np.interp(location,self.r[:-1],c[:-1,10]))
        if getattr(self.args,'backend','native')=='spherical-g1':
            # Finite-mass expansion N=Omega^3 F gives m(S)=-N'''(S)/(12chi^1.5).
            # Expose lower jets: this limit is inadmissible if they do not vanish.
            jet=[float(w@self.mass_numerator[-self.args.width:]) for w in self.mass_jet_weights]
            row.update(mass_numerator_scri=float(self.mass_numerator[-1]),
                mass_numerator_d1_scri=jet[0],mass_numerator_d2_scri=jet[1],
                mass_scri_third_jet=-jet[2]/(12*c[-1,8]**1.5))
        crossings=np.flatnonzero((c[:-1,11]<=0)&(c[1:,11]>0))
        row['apparent_horizon_count']=len(crossings)
        if len(crossings):
            i=crossings[-1];fraction=-c[i,11]/(c[i+1,11]-c[i,11])
            rh=float(self.r[i]+fraction*(self.r[i+1]-self.r[i]));ch=float(c[i,8]+fraction*(c[i+1,8]-c[i,8]))
            radius=rh/(((1-rh*rh)/2)*np.sqrt(ch))
            row.update(apparent_horizon_r=rh,apparent_horizon_areal_radius=radius,
                apparent_horizon_mass=radius/2)
        return row,b,c


class CoupledEvolution(Evolution):
    def parts(self,q):
        import scalar_rhs
        geom=q[:,:8];fields=q[:,8:]
        R,S,diag,beta=super().parts(geom)
        geom=self.geometry_state;d=self.geometry_first;dd=self.geometry_second
        df,ddf=self.op.derivatives(fields,np.ones(2))
        phi,phir,normal=scalar_rhs.matter(self.op,geom,d,fields,df,beta)
        scalar_rhs.add_sources(self.r,geom,R,diag,phir,normal)
        # Use the same stage solve, including its analytically small deviation.
        v=self.shift_deviation
        wave,info=scalar_rhs.rhs(self.op,geom,d,fields,df,ddf,beta,v,
            self.args.kappa1,self.args.kappa2,self.args.theta=='appendix-c',geometry_second=dd,origin_beta_derivative=self.origin_shift_slope)
        self.scalar_info=info;self.scalar_phi=phi;self.scalar_phir=phir;self.scalar_normal=normal
        self.scalar_scri_rate=wave[-1,0]
        self.regularity=np.r_[self.regularity,0.,0.]
        return np.c_[R,wave],np.c_[S,np.zeros_like(wave)],diag,beta

    def report(self,q,t):
        row,b,c=super().report(q,t)
        B=self.scalar_info['potential_numerator'];dB,_=self.op.derivatives(B,1)
        row.update(scalar_max=float(max(abs(self.scalar_phi))),
            scalar_rescaled_scri=float(np.sqrt(1+q[-1,0])*q[-1,8]),
            radius_weighted_scalar_scri=float(q[-1,8]),
            coordinate_time_scalar_flux_scri=float(4*np.pi*self.scalar_scri_rate**2),
            scalar_potential_pole0=float(B[-1]),scalar_potential_pole1=float(dB[-1]))
        return row,b,c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin-grid',action='store_true')
    p.add_argument('--perturbation',choices=['none','constraint','noise'],default='none')
    p.add_argument('--perturbation-amplitude',type=float,default=1e-8)
    p.add_argument('--seed',type=int,default=7419)
    p.add_argument('--origin-variables',action='store_true')
    p.add_argument('--scalar',action='store_true')
    p.add_argument('--scalar-amplitude',type=float,default=.01)
    p.add_argument('--initial-tolerance',type=float,default=1e-10)
    p.add_argument('--origin-ko',type=float,default=0.)
    p.add_argument('--origin-ko-radius',type=float,default=.4)
    p.add_argument('--backend',choices=['native','spherical-g1'],default='native')
    p.add_argument('--quadrature-width',type=int,choices=[5,7,9,11],default=7)
    p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=64);p.add_argument('--t',type=float,default=1.)
    p.add_argument('--cfl',type=float,default=.05);p.add_argument('--amplitude',type=float,default=0.)
    p.add_argument('--pulse',choices=['compact','published-gaussian'],default='compact')
    p.add_argument('--center',type=float,default=.25);p.add_argument('--pulse-width',type=float,default=.1)
    p.add_argument('--kappa1',type=float,default=4.);p.add_argument('--kappa2',type=float,default=-.99)
    p.add_argument('--lapse-restoring',type=float,default=0.)
    p.add_argument('--eta',type=float,default=0.);p.add_argument('--slicing',type=float,default=1.)
    p.add_argument('--theta',choices=['appendix-c','repository'],default='appendix-c')
    p.add_argument('--width',type=int,choices=[5,7,9],default=7)
    p.add_argument('--endpoint',choices=['raw-lhopital','quotient-extrapolation','compatible-trace'],default='raw-lhopital')
    p.add_argument('--regularity-sat',type=float,default=0.)
    p.add_argument('--scri-sat',type=float,default=0.)
    p.add_argument('--report-dt',type=float,default=.1);p.add_argument('--resume',type=Path)
    args=p.parse_args()
    numeric=[args.cfl,args.t,args.amplitude,args.center,args.pulse_width,args.kappa1,args.kappa2,args.eta,args.slicing,args.report_dt,args.origin_ko,args.origin_ko_radius,args.lapse_restoring,args.scalar_amplitude,args.initial_tolerance,args.perturbation_amplitude]
    if not all(np.isfinite(x) for x in numeric):raise ValueError('Nonfinite input')
    if (args.n<16 or args.cfl<=0 or args.t<0 or args.pulse_width<=0 or not 0<=args.center<1
        or args.report_dt<=0 or args.origin_ko<0 or not 0<args.origin_ko_radius<=.75
        or args.kappa1<0 or args.eta<0 or args.slicing<0 or args.lapse_restoring<0
        or args.initial_tolerance<=0):raise ValueError('Invalid input')
    if args.origin_grid and (args.backend!='spherical-g1' or args.origin_variables):raise ValueError('Actual-origin grid uses the analytic g=1 limits with unscaled fields')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    # Bind all actual header dependencies, adapter and Python implementation.
    module_names=['evolve.py','numerics.py','elliptic.py','kernel.py']
    if args.backend=='spherical-g1':module_names+=['g1_rhs.py']
    if args.scalar:module_names+=['scalar_rhs.py','scalar_initial_data.py']
    source=[*sorted((root/'src/z4c/hyperboloidal').glob('*.hpp')),
            *[Path(__file__).with_name(name) for name in module_names],
            Path(__file__).with_name('spherical_api.cpp')]
    binding={'source_sha256':{str(f.relative_to(root)):sha(f) for f in source},
             'executable_sha256':sha(args.library)}
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'command':sys.argv,'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'binding':binding,'checkpoint_parent':None,'status':'running','method':'actual-scri free-Z4c regular-center prototype; coupled scalar optional',
        'resource_authorization':'2026-10-25 user: given the code will be run on 1 spatial dimension, just run the tests locally'}
    if args.scalar and args.backend!='spherical-g1':raise ValueError('Coupled scalar currently requires spherical-g1 backend')
    ev=CoupledEvolution(args) if args.scalar else Evolution(args)
    q=np.zeros((args.n,10 if args.scalar else 8));t=0.;step=0
    # Smooth compact pulse in r^2, exact regular parity, support strictly inside scri.
    x=(ev.r**2-args.center**2)/(2*args.center*args.pulse_width+args.pulse_width**2)
    mask=abs(x)<1
    q[mask,6]=args.amplitude*np.exp(1-1/(1-x[mask]**2))
    if args.pulse=='published-gaussian':
        q[:,6]=args.amplitude*np.exp(-(ev.r**2-args.center**2)**2/(4*args.pulse_width**4))
    if q[-1,6]!=0.:raise ValueError('Initial lapse pulse must vanish at scri for clock-compatible data')
    if args.scalar and not args.resume:
        from scalar_initial_data import solve_data
        initial_grid=ev.r if args.origin_grid else np.r_[0.,ev.r]
        initial=solve_data(len(initial_grid)-1,args.scalar_amplitude,args.initial_tolerance,grid=initial_grid)
        use=slice(None) if args.origin_grid else slice(1,None)
        if not initial['converged']:raise RuntimeError('Initial-data Newton tolerance not achieved; preserve/diagnose before evolution')
        O=ev.omega;w=initial['w'][use];psi=initial['psi'][use]
        q[:,0]=np.expm1(-4*np.log1p(O**3*w))
        phi=initial['phi'][use]
        q[:-1,8]=phi[:-1]/(O[:-1]*np.sqrt(1+q[:-1,0]));q[-1,8]=0.
        q[:-1,9]=q[:-1,8]/O[:-1];q[-1,9]=0.
        manifest['scalar_initial_data']={'amplitude':args.scalar_amplitude,'regular_residual':float(initial['residual']),'Newton_history':initial['iterations'],'Bondi_mass':float(8*w[-1]),'normal_physical_scalar':'n_bar(phi)=0'}
    if args.perturbation!='none' and not args.resume:
        # Compact support strictly inside scri; independent free constraint
        # errors obey center tensor parity. No differential constraint solve.
        rng=np.random.default_rng(args.seed);x=ev.r/.7;envelope=np.zeros_like(x)
        inside=x<1;envelope[inside]=np.exp(1-1/(1-x[inside]**2))
        for field,power in [(0,0),(2,2),(3,0),(4,0),(5,1)]:
            if args.perturbation=='noise':profile=rng.uniform(-1,1,args.n)
            else:profile=sum(rng.normal()*ev.r**(2*k) for k in range(4))
            q[:,field]+=args.perturbation_amplitude*ev.r**power*envelope*profile
    if args.resume:
        with np.load(args.resume,allow_pickle=False) as z:
            old=json.loads(str(z['metadata']))
            if old['binding']!=binding:raise ValueError('Checkpoint source/library binding mismatch')
            for key in ('n','cfl','kappa1','kappa2','eta','slicing','theta','width','endpoint','scri_sat','regularity_sat','lapse_restoring','backend','quadrature_width','origin_ko','origin_ko_radius','scalar','scalar_amplitude','initial_tolerance','origin_variables','origin_grid','perturbation','perturbation_amplitude','seed'):
                if old['arguments'][key]!=manifest['arguments'][key]:raise ValueError('Checkpoint input mismatch: '+key)
            q=z['q'].copy();t=float(z['t']);step=int(z['step'])
        manifest['checkpoint_parent']={'path':str(args.resume.resolve()),'sha256':sha(args.resume)}
    # Preserve exact runnable sources and the matching compiled reference library.
    snapshot=args.output/'frozen-source';snapshot.mkdir()
    for f in source:
        destination=snapshot/f.relative_to(root);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,destination)
        if sha(destination)!=binding['source_sha256'][str(f.relative_to(root))]:raise RuntimeError('Source changed during packaging')
    (snapshot/'runtime').mkdir();shutil.copy2(args.library,snapshot/'runtime'/Path(args.library).name)
    radiated_energy=float(old.get('radiated_energy',0.)) if args.resume else 0.
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.perf_counter();failure=None;history=[];next_report=t
    # Constant CFL until the final landing; resume reproduces aligned timesteps.
    dt=args.cfl*ev.h
    try:
        with (args.output/'history.jsonl').open('w') as log:
            while True:
                if t>=next_report-1e-12 or t>=args.t-1e-12:
                    row,b,c=ev.report(q,t);row['radiated_energy']=radiated_energy;history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    np.savez(args.output/f'profile-{len(history):04d}.npz',r=ev.r,q=q,q_physical=np.c_[ev.geometry_state,q[:,8:]],beta=b,diagnostic=c,t=t)
                    next_report+=args.report_dt
                if t>=args.t-1e-12:break
                h=dt if args.t-t>=dt*(1-1e-12) else args.t-t
                k1=ev.rhs(q);flux1=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                k2=ev.rhs(q+h*k1/2);flux2=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                k3=ev.rhs(q+h*k2/2);flux3=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                k4=ev.rhs(q+h*k3);flux4=4*np.pi*ev.scalar_scri_rate**2 if args.scalar else 0.
                q=q+h*(k1+2*k2+2*k3+k4)/6
                radiated_energy+=h*(flux1+2*flux2+2*flux3+flux4)/6
                step+=1;t+=h
    except Exception as e:
        failure={'type':type(e).__name__,'message':str(e),'time':t,'step':step}
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    finally:
        manifest.update(status='failed' if failure else 'finished',time=t,steps=step,
            runtime_seconds=time.perf_counter()-start,rhs_seconds=ev.rhs_seconds,radiated_energy=radiated_energy,
            elliptic_seconds=ev.solve_seconds,elliptic_calls=ev.calls,failure=failure)
        np.savez(args.output/'checkpoint.npz',q=q,r=ev.r,t=t,step=step,metadata=json.dumps(manifest))
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('status','time','steps','runtime_seconds','failure')}))
    return 1 if failure else 0

if __name__=='__main__':sys.exit(main())
