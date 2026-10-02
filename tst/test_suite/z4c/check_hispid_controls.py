"""Standalone serial exact-geometry checks for the HiSpID pgen and FastFlow.

Uses exported analytic checkpoints, no evolution, and no elliptic solve.
Run with --executable and --manifest (exported data/manifest.json).
"""
import argparse,hashlib,json,math,os,re,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def shape_error(path,lmax,chi):
    # Independent Legendre-polynomial derivatives, not the C++ recurrence.
    coefficients=np.atleast_2d(np.loadtxt(path))[-1]
    if coefficients.size!=(lmax+1)**2:raise ValueError('wrong horizon coefficient count')
    mu,_=np.polynomial.legendre.leggauss(40)
    phi=2*np.pi*(np.arange(80)+.37)/80
    mu,phi=np.meshgrid(mu,phi);mu=mu.ravel();phi=phi.ravel();radius=np.zeros(len(mu));index=0
    for l in range(lmax+1):
        polynomial=np.zeros(l+1);polynomial[l]=1
        for m in range(l+1):
            derivative=np.polynomial.legendre.legder(polynomial,m)
            polar=(-1)**m*(1-mu*mu)**(m/2)*np.polynomial.legendre.legval(mu,derivative)
            norm=np.sqrt((2*l+1)/(4*np.pi))*np.exp(.5*(math.lgamma(l-m+1)-math.lgamma(l+m+1)))
            if m==0:radius+=coefficients[index]*norm*polar;index+=1
            else:
                radius+=np.sqrt(2)*norm*polar*(coefficients[index]*np.cos(m*phi)+coefficients[index+1]*np.sin(m*phi));index+=2
    nx=np.sqrt(1-mu*mu)*np.cos(phi);G=1/np.sqrt(1-.885**2)
    expected=.5*np.sqrt(1-chi*chi)/np.sqrt(1+(G*G-1)*nx*nx)
    return float(np.max(abs(radius/expected-1)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cases',default='flat,schwarzschild,kerr95,boost885,kerr95_boost885')
    p.add_argument('--boost-levels',default='16,24,32,48')
    p.add_argument('--combined-flow-alpha',type=float,default=.2)
    a=p.parse_args();exe=Path(a.executable).resolve();manifest=json.loads(Path(a.manifest).read_text())
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    evidence={'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'initial_time':0,'evolution_steps':0,'records':[],'passed':False}
    cases=[]
    for case in a.cases.split(','):
        if case=='flat':cases.append((case,8,16,1.0))
        elif case=='schwarzschild':cases.extend((case,8,16,s) for s in (.8,1.2))
        elif case=='kerr95':cases.extend((case,l,n,s) for l,n,s in ((8,16,.8),(12,24,1.2),(16,32,1.05)))
        else:cases.extend((case,l,l+2,1.02) for l in map(int,a.boost_levels.split(',')))
    for case,lmax,ntheta,scale in cases:
        source=manifest['schwarzschild' if case=='flat' else case]
        run=root/f'{case}_l{lmax}_n{ntheta}_s{scale}';run.mkdir(exist_ok=False)
        cmd=[str(exe),'-i',str(ROOT/'inputs/hispid.athinput'),
             'problem/hispid_filename='+source['path'],'problem/hispid_source_sha256='+source['source_library_sha256'],
             f'fastflow/lmax={lmax}',f'fastflow/ntheta={ntheta}',f'problem/hispid_horizon_guess_scale={scale}']
        if case=='flat':
            cmd+=['problem/hispid_flat_control=true','problem/hispid_seed_horizon_guess=false',
                  'fastflow/initial_radius_0=4','fastflow/mass_tol_0=100','fastflow/expansion_rms_tol_0=.6']
            cmd+=['fastflow/hmean_tol_0=1000']
            for d in (1,2,3):cmd.extend([f'mesh/x{d}min=-5',f'mesh/x{d}max=5'])
        elif 'boost885' in case and lmax<max(map(int,a.boost_levels.split(','))):
            # Underresolved coarse searches are diagnostics; only the finest
            # is required to satisfy the strict expansion gate below.
            cmd+=['fastflow/expansion_rms_tol_0=.1']
        if case=='kerr95_boost885':
            cmd+=[f'fastflow/flow_alpha_beta_const_0={a.combined_flow_alpha}','fastflow/flow_iterations_0=3000']
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        start=time.monotonic();r=subprocess.run(cmd,cwd=run,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
        (run/'run.log').write_text(r.stdout)
        row={'case':case,'lmax':lmax,'ntheta':ntheta,'initial_scale':scale,'returncode':r.returncode,'seconds':time.monotonic()-start,'command':cmd,'source':source,'passed':False}
        row['flow_alpha']=a.combined_flow_alpha if case=='kerr95_boost885' else 1.0
        row['zero_evolution_verified']=bool(re.search(r'time=0\.000000e\+00 cycle=0',r.stdout) and 'MeshBlock-cycles = 0' in r.stdout)
        match=re.search(r'attempt_area=([\deE+.-]+) attempt_expansion_rms=([\deE+.-]+)',r.stdout)
        if match:row.update(attempt_area=float(match[1]),attempt_expansion_rms=float(match[2]))
        path=run/'hispid.horizon_summary_0.txt'
        if r.returncode==0 and path.exists():
            values=np.atleast_2d(np.loadtxt(path))[-1]
            # n,time,mass,Sx,Sy,Sz,S,area,mean_square_expansion,hmean,rmean,rmin.
            row.update(summary=values.tolist(),area=float(values[7]),expansion_rms=float(np.sqrt(values[8])),min_radius=float(values[11]))
            chi=.95 if 'kerr95' in case else 0
            expected_area=64*np.pi if case=='flat' else 8*np.pi*(1+np.sqrt(1-chi*chi))
            row['expected_area']=float(expected_area);row['relative_area_error']=float(abs(values[7]/expected_area-1))
            if case=='flat':
                row['passed']=bool(np.isfinite(values).all() and row['relative_area_error']<1e-12 and abs(values[8]-.25)<1e-12 and np.max(abs(values[3:7]))<1e-12)
            else:
                row['passed']=bool(np.isfinite(values).all() and row['relative_area_error']<2e-5 and row['expansion_rms']<1e-7 and row['min_radius']>0)
                if case=='kerr95':
                    row['spin_error']=float(abs(values[5]-.95));row['passed'] &= row['spin_error']<2e-5
                elif 'boost885' in case:
                    row['shape_sampled_relative_linf']=shape_error(run/'hispid.horizon_shape_0.txt',lmax,chi)
                    row['passed'] &= row['shape_sampled_relative_linf']<1e-6
            row['passed'] &= row['zero_evolution_verified']
        evidence['records'].append(row)
        complete=len(evidence['records'])==len(cases)
        groups={name:[x for x in evidence['records'] if x['case']==name] for name in a.cases.split(',')}
        evidence['passed']=bool(complete and all(
            rows[-1]['passed'] and len(rows)>=3 and all(rows[i+1].get('expansion_rms',np.inf)<rows[i].get('expansion_rms',0) for i in range(len(rows)-1))
            if 'boost885' in name else all(x['passed'] for x in rows)
            for name,rows in groups.items()))
        (root/'controls.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(case,lmax,ntheta,'passed',row['passed'],'seconds',row['seconds'],flush=True)
        if r.returncode:print(r.stdout[-2500:],flush=True)
    return 0 if evidence['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
