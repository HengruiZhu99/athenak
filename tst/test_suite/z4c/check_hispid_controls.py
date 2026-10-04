"""Standalone serial exact-geometry checks for the HiSpID pgen and FastFlow.

Uses exported analytic checkpoints, no evolution, and no elliptic solve.
Run with --executable and --manifest (exported data/manifest.json).
"""
import argparse,hashlib,json,math,os,re,subprocess,time
from pathlib import Path
import numpy as np
from check_hispid_binary import checkpoint_metadata
from hispid_sampler_proof import validate_migration,import_evidence
from fastflow_storage import harmonic_table_bytes,storage_evidence

ROOT=Path(__file__).resolve().parents[2]

def shape_error(path,lmax,chi,speed=.885):
    # Independent Legendre polynomials; high orders use SciPy's normalized
    # harmonic oracle to avoid overflowing unnormalized factorial derivatives.
    coefficients=np.atleast_2d(np.loadtxt(path))[-1]
    if coefficients.size!=(lmax+1)**2:raise ValueError('wrong horizon coefficient count')
    nt,nphi=max(40,2*lmax+3),max(80,4*lmax+6)
    mu,_=np.polynomial.legendre.leggauss(nt);mu=np.r_[mu,-1.,0.,1.]
    phi=np.r_[2*np.pi*(np.arange(nphi)+.37)/nphi,np.arange(4)*np.pi/2]
    cosine=np.zeros((len(mu),lmax+1));sine=cosine.copy();index=0
    if lmax>64:
        from scipy.special import sph_harm_y
    for l in range(lmax+1):
        polynomial=np.zeros(l+1);polynomial[l]=1
        for m in range(l+1):
            if lmax>64:polar=sph_harm_y(l,m,np.arccos(mu),0).real
            else:
                derivative=np.polynomial.legendre.legder(polynomial,m)
                polar=(-1)**m*(1-mu*mu)**(m/2)*np.polynomial.legendre.legval(mu,derivative)
                norm=np.sqrt((2*l+1)/(4*np.pi))*np.exp(.5*(math.lgamma(l-m+1)-math.lgamma(l+m+1)))
                polar*=norm
            if m==0:cosine[:,m]+=coefficients[index]*polar;index+=1
            else:
                cosine[:,m]+=np.sqrt(2)*coefficients[index]*polar
                sine[:,m]+=np.sqrt(2)*coefficients[index+1]*polar;index+=2
    radius=cosine[:,0,None]+np.zeros((len(mu),len(phi)))
    for m in range(1,lmax+1):radius+=cosine[:,m,None]*np.cos(m*phi)+sine[:,m,None]*np.sin(m*phi)
    nx=np.sqrt(1-mu*mu)[:,None]*np.cos(phi);G=1/np.sqrt(1-speed**2)
    expected=.5*np.sqrt(1-chi*chi)/np.sqrt(1+(G*G-1)*nx*nx)
    if not np.isfinite(radius).all():raise ValueError('nonfinite independent surface oracle')
    return float(np.max(abs(radius/expected-1)))


def refinement_qualified(rows,boosted):
    if not rows or not all(row.get('returncode')==0 and row.get('bound_inputs_unchanged') is True
        and row.get('zero_evolution_verified') is True and row.get('import',{}).get('passed') is True
        and ('harmonic_storage' not in row or row.get('harmonic_allocation',{}).get('passed') is True) for row in rows):
        return False
    if boosted:
        return bool(len(rows)>=3 and rows[-1]['passed'] and all(
            rows[i+1].get('expansion_rms',np.inf)<rows[i].get('expansion_rms',0) for i in range(len(rows)-1)))
    return all(row['passed'] for row in rows)


def seed_targets(case):
    return (.99 if case=='kerr99' else .95 if 'kerr95' in case else 0.,
            float(np.sqrt(.99)) if case=='gamma10' else .885 if 'boost885' in case else 0.)


def boost_orders(specification,cases,diagnostic_single=False):
    levels=list(map(int,specification.split(',')))
    if any(l<2 for l in levels) or any(x>=y for x,y in zip(levels,levels[1:])):
        raise ValueError('strictly increasing distinct boost harmonic orders >=2 required')
    if diagnostic_single:
        if cases!=['gamma10'] or len(levels)!=1:
            raise ValueError('single-level diagnostic requires only gamma10 and exactly one boost order')
    elif len(levels)<3:
        raise ValueError('three or more strictly increasing distinct boost harmonic orders required')
    return levels


def exact_seed_source(entry,case):
    """The case name cannot substitute for the actual analytic configuration."""
    path=Path(entry['path']).resolve(strict=True);source=checkpoint_metadata(path,False)
    if any(source[key]!=entry[key] for key in ('file_sha256','source_library_sha256','acceptance')):
        raise ValueError('manifest/checkpoint binding differs')
    if source['acceptance']!='analytic_seed':raise ValueError('exact analytic seed checkpoint required')
    chi,speed=seed_targets(case);hole=source['holes'][0]
    expected=np.r_[1.,np.zeros(3),[0.,0.,chi],[speed,0.,0.]]
    if not np.array_equal(hole,expected) or source['holes'][1][0]!=0:
        raise ValueError('exact seed parameters differ from the declared control')
    header={};values=[];terminated=False
    with path.open() as stream:
        for line in stream:
            words=line.split()
            if words[0]=='unknowns':count=int(words[1]);break
            header[words[0]]=words[1:]
        for line in stream:
            if line.split()==['END']:terminated=True;break
            values.extend(map(float,line.split()))
        if stream.read().strip():raise ValueError('trailing exact-control data')
    if (not terminated or len(values)!=count or count!=4*int(np.prod(list(map(int,header['n']))))
        or not np.isfinite(values).all() or np.any(np.asarray(values)!=0)
        or any(float(x)!=0 for key in ('omega','inner_min','inner_max','far_radius','inner_flatten','conformal_choice') for x in header[key])):
        raise ValueError('exact controls require zero corrections and unmodified seed geometry')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['file_sha256']:
        raise ValueError('checkpoint changed during exact-control decoding')
    return source

def main():
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cases',default='flat,schwarzschild,kerr95,boost885,kerr95_boost885')
    p.add_argument('--boost-levels',default='16,24,32,48')
    p.add_argument('--diagnostic-single-boost-level',action='store_true',help='one Gamma10 finder diagnosis only; cannot qualify refinement or aggregate acceptance')
    p.add_argument('--combined-flow-alpha',type=float,default=.2)
    p.add_argument('--boost-flow-alpha',type=float,default=1.,help='step factor for the separate Gamma10 control')
    p.add_argument('--flow-iterations',type=int,default=3000,help='iteration cap for Gamma10/combined-boost controls; does not change acceptance')
    p.add_argument('--worker-timeout',type=float,default=900.,help='bounded worker time in seconds; partial evidence remains unqualified')
    p.add_argument('--full-precision-trace',action='store_true',help='retain all finder iterates at round-trip precision')
    p.add_argument('--consumer-memory-mib',type=int,default=32768,help='explicit Serial consumer screen: harmonic tables plus1GiB allowance; actual memory is separate')
    p.add_argument('--harmonic-storage',choices=('dense','factorized'),default='dense')
    a=p.parse_args();exe=Path(a.executable).resolve(strict=True)
    manifest_path=Path(a.manifest).resolve(strict=True);manifest_bytes=manifest_path.read_bytes()
    manifest_sha=hashlib.sha256(manifest_bytes).hexdigest();manifest=json.loads(manifest_bytes)
    if any(not np.isfinite(x) or x<=0 for x in (a.combined_flow_alpha,a.boost_flow_alpha)):
        raise ValueError('positive finite flow step factors required')
    if a.flow_iterations<=0 or not np.isfinite(a.worker_timeout) or a.worker_timeout<=0:
        raise ValueError('positive finder iteration cap and finite worker timeout required')
    case_names=a.cases.split(',')
    boost_levels=boost_orders(a.boost_levels,case_names,a.diagnostic_single_boost_level)
    if a.consumer_memory_mib<2048:raise ValueError('consumer screen needs at least2048MiB')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'controls.json').exists():raise FileExistsError('preserve prior exact controls in a separate output directory')
    template=(ROOT/'inputs/hispid.athinput').read_text()
    evidence={'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
        'manifest_sha256':manifest_sha,'input_template_sha256':hashlib.sha256(template.encode()).hexdigest(),
        'initial_time':0,'evolution_steps':0,'records':[],'passed':False}
    evidence['harmonic_storage']=a.harmonic_storage
    evidence['diagnostic_single_boost_level']=a.diagnostic_single_boost_level
    evidence['worker_timeout_seconds']=a.worker_timeout
    evidence['full_precision_trace']=a.full_precision_trace
    script_paths=[Path(__file__).resolve(),*(Path(__file__).resolve().parent/name for name in
        ('check_hispid_binary.py','hispid_sampler_proof.py','fastflow_storage.py'))]
    evidence['workflow_source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in script_paths}
    cases=[]
    for case in case_names:
        if case=='flat':cases.append((case,8,16,1.0))
        elif case=='schwarzschild':cases.extend((case,8,16,s) for s in (.8,1.2))
        elif case in ('kerr95','kerr99'):cases.extend((case,l,n,s) for l,n,s in ((8,16,.8),(12,24,1.2),(16,32,1.05)))
        elif case in ('boost885','kerr95_boost885','gamma10'):cases.extend((case,l,l+2,1.02) for l in boost_levels)
        else:raise ValueError('unsupported exact control: '+case)
    estimates=[dict(case=case,lmax=l,ntheta=nt,harmonic_table_bytes=harmonic_table_bytes(l,nt,a.harmonic_storage)) for case,l,nt,_ in cases]
    if any(row['harmonic_table_bytes']+1024**3>a.consumer_memory_mib*1024**2 for row in estimates):
        raise ValueError('declared harmonic-table storage plus allowance exceeds the Serial consumer budget')
    evidence['consumer_memory_screen']=dict(budget_mib=a.consumer_memory_mib,other_allowance_bytes=1024**3,
        estimates=estimates,note='Single-copy Serial harmonic data only, plus an explicit allowance; not a measured peak or a complete allocation guarantee.')
    for case,lmax,ntheta,scale in cases:
        entry=manifest['schwarzschild' if case=='flat' else case]
        source=exact_seed_source(entry,'schwarzschild' if case=='flat' else case)
        migration=validate_migration(entry['migration_proof'],source) if entry.get('migration_proof') else None
        dependencies=entry.get('library_dependency_images',{})
        if case in ('kerr99','gamma10') and not migration:
            raise ValueError('fresh extreme controls require a separate-process pure-reference-consumer sampler proof')
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in dependencies.items()):
            raise ValueError('exact-control dependency changed before worker')
        if (hashlib.sha256(exe.read_bytes()).hexdigest()!=evidence['executable_sha256']
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=manifest_sha
            or any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in evidence['workflow_source_sha256'].items())):
            raise ValueError('bound executable/manifest/workflow changed before exact horizon worker')
        run=root/f'{case}_l{lmax}_n{ntheta}_s{scale}';run.mkdir(exist_ok=False)
        input_path=run/'control.athinput';input_path.write_text(template)
        input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
        cmd=[str(exe),'-i',str(input_path),
             'problem/hispid_filename='+source['path'],'problem/hispid_source_sha256='+source['source_library_sha256'],
             'fastflow/factorized_harmonics='+str(a.harmonic_storage=='factorized').lower(),
             f'fastflow/lmax={lmax}',f'fastflow/ntheta={ntheta}',f'problem/hispid_horizon_guess_scale={scale}']
        if case=='flat':
            cmd+=['problem/hispid_flat_control=true','problem/hispid_seed_horizon_guess=false',
                  'fastflow/initial_radius_0=4','fastflow/mass_tol_0=100','fastflow/expansion_rms_tol_0=.6']
            cmd+=['fastflow/hmean_tol_0=1000']
            for d in (1,2,3):cmd.extend([f'mesh/x{d}min=-5',f'mesh/x{d}max=5'])
        elif ('boost885' in case or case=='gamma10') and lmax<max(map(int,a.boost_levels.split(','))):
            # Underresolved coarse searches are diagnostics; only the finest
            # is required to satisfy the strict expansion gate below.
            cmd+=['fastflow/expansion_rms_tol_0=.1']
        if case=='kerr95_boost885':
            cmd+=[f'fastflow/flow_alpha_beta_const_0={a.combined_flow_alpha}',f'fastflow/flow_iterations_0={a.flow_iterations}']
        if case=='gamma10':cmd+=[f'fastflow/flow_alpha_beta_const_0={a.boost_flow_alpha}',f'fastflow/flow_iterations_0={a.flow_iterations}']
        if a.full_precision_trace:cmd+=['fastflow/full_precision_trace=true']
        if migration:cmd+=['problem/hispid_allow_library_migration=true']
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        start=time.monotonic()
        with (run/'run.log').open('w') as log:
            try:r=subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.worker_timeout)
            except subprocess.TimeoutExpired:returncode='timeout'
            else:returncode=r.returncode
        stdout=(run/'run.log').read_text()
        row={'case':case,'lmax':lmax,'ntheta':ntheta,'initial_scale':scale,'returncode':returncode,'seconds':time.monotonic()-start,'command':cmd,'source':source,'passed':False,
            'input_sha256':input_sha,'sampler_migration':migration}
        row['harmonic_storage']=a.harmonic_storage
        row['harmonic_allocation']=storage_evidence(stdout,a.harmonic_storage,lmax,ntheta)
        evidence['records'].append(row);evidence['passed']=False
        (root/'controls.json').write_text(json.dumps(evidence,indent=2)+'\n')
        row['flow_alpha']=a.combined_flow_alpha if case=='kerr95_boost885' else a.boost_flow_alpha if case=='gamma10' else 1.0
        if case in ('gamma10','kerr95_boost885'):row['flow_iteration_limit']=a.flow_iterations
        unchanged=bool(hashlib.sha256(exe.read_bytes()).hexdigest()==evidence['executable_sha256']
            and hashlib.sha256(input_path.read_bytes()).hexdigest()==input_sha
            and hashlib.sha256(manifest_path.read_bytes()).hexdigest()==manifest_sha
            and hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['file_sha256']
            and all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in evidence['workflow_source_sha256'].items()))
        if migration:unchanged &= validate_migration(entry['migration_proof'],source)==migration
        row['bound_inputs_unchanged']=unchanged;row['import']=import_evidence(stdout,source,migration)
        if not migration and dependencies:
            row['import']['passed'] &= row['import'].get('consumer_dependency_images')==dependencies
            unchanged &= all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in dependencies.items())
            row['bound_inputs_unchanged']=unchanged
        row['zero_evolution_verified']=bool(re.search(r'time=0\.000000e\+00 cycle=0',stdout) and 'MeshBlock-cycles = 0' in stdout)
        match=re.search(r'attempt_area=([\deE+.-]+) attempt_expansion_rms=([\deE+.-]+)',stdout)
        if match:row.update(attempt_area=float(match[1]),attempt_expansion_rms=float(match[2]))
        path=run/'hispid.horizon_summary_0.txt'
        if returncode==0 and path.exists():
            values=np.atleast_2d(np.loadtxt(path))[-1]
            # n,time,mass,Sx,Sy,Sz,S,area,mean_square_expansion,hmean,rmean,rmin.
            row.update(summary=values.tolist(),area=float(values[7]),expansion_rms=float(np.sqrt(values[8])),min_radius=float(values[11]))
            chi,speed=seed_targets(case)
            expected_area=64*np.pi if case=='flat' else 8*np.pi*(1+np.sqrt(1-chi*chi))
            row['expected_area']=float(expected_area);row['relative_area_error']=float(abs(values[7]/expected_area-1))
            if case=='flat':
                row['passed']=bool(np.isfinite(values).all() and row['relative_area_error']<1e-12 and abs(values[8]-.25)<1e-12 and np.max(abs(values[3:7]))<1e-12)
            else:
                row['passed']=bool(np.isfinite(values).all() and row['relative_area_error']<2e-5 and row['expansion_rms']<1e-7 and row['min_radius']>0)
                if case in ('kerr95','kerr99'):
                    row['spin_error']=float(abs(values[5]-chi));row['passed'] &= row['spin_error']<2e-5
                elif 'boost885' in case or case=='gamma10':
                    row['shape_sampled_relative_linf']=shape_error(run/'hispid.horizon_shape_0.txt',lmax,chi,speed)
                    row['passed'] &= row['shape_sampled_relative_linf']<1e-6
                    row['shape_oracle']=dict(method='scipy.special.sph_harm_y' if lmax>64 else 'NumPy Legendre-polynomial derivatives',
                        ntheta=max(40,2*lmax+3),nphi=max(80,4*lmax+6),additional_cardinal_directions=True,
                        continuous_shape_certificate=False)
                row['expected_irreducible_mass']=float(np.sqrt(expected_area/(16*np.pi)))
                row['expected_horizon_mass']=1.;row['horizon_mass_error']=float(abs(values[2]-1))
                row['expected_seed_lorentz_factor']=float(1/np.sqrt(1-speed**2))
                row['passed'] &= row['horizon_mass_error']<2e-5
                if chi==0:row['passed'] &= np.max(abs(values[3:7]))<2e-5
            row['passed'] &= row['zero_evolution_verified'] and unchanged and row['import']['passed'] and row['harmonic_allocation']['passed']
        complete=len(evidence['records'])==len(cases)
        groups={name:[x for x in evidence['records'] if x['case']==name] for name in a.cases.split(',')}
        evidence['case_qualification']={name:bool(len(rows)==sum(c[0]==name for c in cases)
            and refinement_qualified(rows,'boost885' in name or name=='gamma10')) for name,rows in groups.items()}
        evidence['passed']=bool(complete and all(evidence['case_qualification'].values()))
        (root/'controls.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(case,lmax,ntheta,'passed',row['passed'],'seconds',row['seconds'],flush=True)
        if returncode:print(stdout[-2500:],flush=True)
        if not unchanged:raise ValueError('bound inputs changed; retained control stays unqualified')
    return 0 if evidence['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
