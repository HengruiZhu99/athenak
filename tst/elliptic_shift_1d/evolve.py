#!/usr/bin/env python3
"""Experimental actual-scri free-Z4c vacuum driver; all failures are retained.

Regular-center grid only. No puncture or coupled-matter claims.
"""
import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
from elliptic import solve,residual,connection
from numerics import RadialOperators,weights
from kernel import Kernel,PARITY


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evolution:
    def __init__(self,args):
        self.args=args;self.r=(np.arange(args.n)+.5)/(args.n-.5)
        self.h=1/(args.n-.5);self.op=RadialOperators(self.r,width=args.width)
        self.kernel=Kernel(args.library);self.calls=0;self.solve_seconds=0.
        self.rhs_seconds=0.;self.elliptic_differential_residual=0.
        self.omega=(1-self.r*self.r)/2
        self.scri_quotient_weights=weights(self.r[-args.width:-1],1.)

    def parts(self,q):
        d,dd=self.op.derivatives(q,PARITY)
        a=1+q[:,1];chi=1+q[:,0];alpha=(1+self.r*self.r)/2+q[:,6]
        if min(np.min(a),np.min(chi),np.min(alpha))<=0:
            raise FloatingPointError('Nonpositive metric or lapse')
        start=time.perf_counter()
        beta,db,ddb,Q=solve(self.op,a,d[:,1],dd[:,1],alpha,self.r+d[:,6],
            q[:,2],d[:,2],-alpha[-1]*np.sqrt(chi[-1]/a[-1]),eta=self.args.eta)
        self.solve_seconds+=time.perf_counter()-start;self.calls+=1
        R,S,diag=self.kernel.parts(self.r,q,d,dd,beta,db,ddb,self.args.kappa1,
            self.args.kappa2,self.args.theta=='appendix-c',self.args.slicing)
        # Independent sampled-shift residual; do not use the analytic derivatives
        # returned by solve() for this diagnostic.
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
        result=R.copy();result[:-1]+=S[:-1]/self.omega[:-1,None]
        dS,_=self.op.derivatives(S,PARITY)
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
            drate,_=self.op.derivatives(result,PARITY)
            result[-1,2]=drate[-1,1]/6-2*drate[-1,0]/3
        # Weak residual penalty; vanishes identically on the regular continuum
        # phase. It damps boundary incompatibility without resetting any field.
        result[-1]+=self.args.scri_sat/self.h*S[-1]
        result[-1]-=self.args.regularity_sat/self.h*self.regularity
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
        row={'t':float(t),'max_field':float(np.max(np.abs(q[:,:7]))),
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
        return row,b,c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=64);p.add_argument('--t',type=float,default=1.)
    p.add_argument('--cfl',type=float,default=.05);p.add_argument('--amplitude',type=float,default=0.)
    p.add_argument('--pulse',choices=['compact','published-gaussian'],default='compact')
    p.add_argument('--center',type=float,default=.25);p.add_argument('--pulse-width',type=float,default=.1)
    p.add_argument('--kappa1',type=float,default=4.);p.add_argument('--kappa2',type=float,default=-.99)
    p.add_argument('--eta',type=float,default=0.);p.add_argument('--slicing',type=float,default=1.)
    p.add_argument('--theta',choices=['appendix-c','repository'],default='appendix-c')
    p.add_argument('--width',type=int,choices=[5,7,9],default=7)
    p.add_argument('--endpoint',choices=['raw-lhopital','quotient-extrapolation','compatible-trace'],default='raw-lhopital')
    p.add_argument('--regularity-sat',type=float,default=0.)
    p.add_argument('--scri-sat',type=float,default=0.)
    p.add_argument('--report-dt',type=float,default=.1);p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if args.n<16 or args.cfl<=0 or args.t<0 or args.pulse_width<=0:raise ValueError('Invalid input')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    # Bind all actual header dependencies, adapter and Python implementation.
    source=[*sorted((root/'src/z4c/hyperboloidal').glob('*.hpp')),
            *sorted(Path(__file__).resolve().parent.glob('*.py')),
            Path(__file__).with_name('spherical_api.cpp')]
    binding={'source_sha256':{str(f.relative_to(root)):sha(f) for f in source},
             'executable_sha256':sha(args.library)}
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'command':sys.argv,'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'binding':binding,'checkpoint_parent':None,'status':'running','method':'actual-scri vacuum regular-center prototype v1',
        'resource_authorization':'2026-10-25 user: given the code will be run on 1 spatial dimension, just run the tests locally'}
    ev=Evolution(args);q=np.zeros((args.n,8));t=0.;step=0
    # Smooth compact pulse in r^2, exact regular parity, support strictly inside scri.
    x=(ev.r**2-args.center**2)/(2*args.center*args.pulse_width+args.pulse_width**2)
    mask=abs(x)<1
    q[mask,6]=args.amplitude*np.exp(1-1/(1-x[mask]**2))
    if args.pulse=='published-gaussian':
        q[:,6]=args.amplitude*np.exp(-(ev.r**2-args.center**2)**2/(4*args.pulse_width**4))
        if q[-1,6]!=0.:raise ValueError('Published Gaussian must vanish to machine precision at scri for clock-compatible data')
    if args.resume:
        with np.load(args.resume,allow_pickle=False) as z:
            old=json.loads(str(z['metadata']))
            if old['binding']!=binding:raise ValueError('Checkpoint source/library binding mismatch')
            for key in ('n','cfl','kappa1','kappa2','eta','slicing','theta','width','endpoint','scri_sat','regularity_sat'):
                if old['arguments'][key]!=manifest['arguments'][key]:raise ValueError('Checkpoint input mismatch: '+key)
            q=z['q'].copy();t=float(z['t']);step=int(z['step'])
        manifest['checkpoint_parent']={'path':str(args.resume.resolve()),'sha256':sha(args.resume)}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.perf_counter();failure=None;history=[];next_report=t
    # Constant CFL until the final landing; resume reproduces aligned timesteps.
    dt=args.cfl*ev.h
    try:
        with (args.output/'history.jsonl').open('w') as log:
            while True:
                if t>=next_report-1e-12 or t>=args.t-1e-12:
                    row,b,c=ev.report(q,t);history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    np.savez(args.output/f'profile-{len(history):04d}.npz',r=ev.r,q=q,beta=b,diagnostic=c,t=t)
                    next_report+=args.report_dt
                if t>=args.t-1e-12:break
                h=dt if args.t-t>=dt*(1-1e-12) else args.t-t
                k1=ev.rhs(q);k2=ev.rhs(q+h*k1/2);k3=ev.rhs(q+h*k2/2);k4=ev.rhs(q+h*k3)
                q=q+h*(k1+2*k2+2*k3+k4)/6
                step+=1;t+=h
    except Exception as e:
        failure={'type':type(e).__name__,'message':str(e),'time':t,'step':step}
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    finally:
        manifest.update(status='failed' if failure else 'finished',time=t,steps=step,
            runtime_seconds=time.perf_counter()-start,rhs_seconds=ev.rhs_seconds,
            elliptic_seconds=ev.solve_seconds,elliptic_calls=ev.calls,failure=failure)
        np.savez(args.output/'checkpoint.npz',q=q,t=t,step=step,metadata=json.dumps(manifest))
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('status','time','steps','runtime_seconds','failure')}))
    return 1 if failure else 0

if __name__=='__main__':sys.exit(main())
