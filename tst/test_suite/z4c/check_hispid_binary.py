"""Initial-time solved-binary FastFlow refinement and inner-ball enclosure.

Uses direct native geometry. Mesh round trips are checked by the pgen; this
is not a mesh-resolved horizon or production evolution test.
"""
import argparse,hashlib,json,math,os,re,subprocess,time
from pathlib import Path
import numpy as np
from hispid_sampler_proof import validate_migration,import_evidence
from fastflow_storage import harmonic_table_bytes,storage_evidence

ROOT=Path(__file__).resolve().parents[2]


def schedule_prerequisites(rows,expected):
    """Coarse expansion is diagnostic, but every execution witness is required."""
    return bool(len(rows)==expected and all(r.get('returncode')==0
        and r.get('bound_inputs_unchanged') is True and r.get('zero_evolution_verified') is True
        and r.get('surface_kind_verified') is True and r.get('import',{}).get('passed') is True
        and r.get('harmonic_allocation',{}).get('passed') is True
        and r.get('shape_guess_verified',True) is True
        and r.get('parallel_geometry_verified',True) is True for r in rows))


def uniform_bound(coefficients):
    """Addition theorem and Cauchy-Schwarz for orthonormal real harmonics."""
    c=np.asarray(coefficients);lmax=math.isqrt(len(c))-1
    if (lmax+1)**2!=len(c) or not np.isfinite(c).all():raise ValueError('invalid surface coefficients')
    return float(sum(np.linalg.norm(c[l*l:(l+1)**2])*math.sqrt((2*l+1)/(4*math.pi)) for l in range(lmax+1)))


def radius_lower_bound(c):
    c=np.asarray(c)
    return float(c[0]/math.sqrt(4*math.pi)-uniform_bound(np.r_[0.,c[1:]]))


def shape_change(a,b):
    n=max(len(a),len(b));aa=np.zeros(n);bb=np.zeros(n);aa[:len(a)]=a;bb[:len(b)]=b
    return uniform_bound(aa-bb)


def checkpoint_metadata(path,require_two_active=True):
    initial_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    meta={}
    with path.open() as f:
        for line in f:
            words=line.split()
            if not words:continue
            if words[0]=='unknowns':break
            if words[0] in meta:raise ValueError('duplicate checkpoint field')
            meta[words[0]]=words[1:]
    version=meta.get('HISPID_CHECKPOINT')
    if version not in (['1'],['2']):raise ValueError('checkpoint version')
    family='qi'
    if version==['2']:
        if meta.get('seed_family') not in (['qi'],['trumpet_r0_m']):raise ValueError('checkpoint seed family')
        family=meta['seed_family'][0]
    elif 'seed_family' in meta:raise ValueError('v1 checkpoint cannot carry a seed family')
    holes=[list(map(float,meta['hole'+str(h)])) for h in range(2)]
    if require_two_active and any(x[0]<=0 for x in holes):raise ValueError('two active holes required')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=initial_sha:raise ValueError('checkpoint changed while reading metadata')
    return dict(path=str(path),file_sha256=initial_sha,
                source_library_sha256=meta['library_sha256'][0],acceptance=meta['acceptance'][0],
                holes=holes,inner_max=list(map(float,meta['inner_max'])),
                inner_flatten=int(meta['inner_flatten'][0]),parameterization=meta['parameterization'][0],seed_family=family)


def main():
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--allow-diagnostic',action='store_true',help='measure horizons of explicitly labeled unvalidated data without promoting its constraint acceptance')
    p.add_argument('--geometry-threads',type=int,default=1);p.add_argument('--strict-expansion',action='store_true',help='require expansion RMS below1e-7 at every angular order');p.add_argument('--output',required=True);p.add_argument('--levels',default='8,12,16')
    p.add_argument('--flow-alpha',type=float,default=1.,help='positive FastFlow step-size factor; does not change acceptance tolerances')
    p.add_argument('--guess-scale',type=float,default=1.05,help='positive multiplier of the seed horizon radius for the initial guess')
    p.add_argument('--initial-shapes',nargs='+',help='one retained coefficient file per surface, used only as the first initial guess')
    p.add_argument('--reuse-shapes',action='store_true',help='initialize each later order from the preceding passed surface; recompute all checks')
    p.add_argument('--flow-iterations',type=int,default=600,help='positive maximum per-surface iteration count; acceptance tolerances stay unchanged')
    p.add_argument('--domain-half-width',type=float,default=8.,help='positive mesh-domain half width; every active hole and modified ball must fit')
    p.add_argument('--migration-proof',help='checkpoint-bound, separate-process CPU sampler proof; no physical acceptance transfer')
    p.add_argument('--harmonic-storage',choices=('dense','factorized'),default='dense')
    p.add_argument('--consumer-memory-mib',type=int,default=32768)
    p.add_argument('--enclosure-axis',choices=('x','y','z'),help='optional outward interval range of the retained SH surface about this Cartesian axis')
    p.add_argument('--enclosure-intervals',type=int,default=1024)
    p.add_argument('--common',action='store_true',help='separate common-surface search; a failed search does not establish absence')
    p.add_argument('--common-center',default='0,0,0',help='explicit Cartesian common-search center')
    p.add_argument('--common-radius',type=float,help='positive initial common-search sphere radius')
    p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    if a.geometry_threads<1:raise ValueError('positive geometry thread count required')
    if not math.isfinite(a.flow_alpha) or a.flow_alpha<=0:raise ValueError('positive finite flow alpha required')
    if (a.enclosure_axis and (a.enclosure_intervals<16 or a.enclosure_intervals>16384 or a.enclosure_intervals%2)):
        raise ValueError('even enclosure interval count in[16,16384] required')
    if not a.enclosure_axis and a.enclosure_intervals!=1024:raise ValueError('enclosure intervals require an enclosure axis')
    if not math.isfinite(a.guess_scale) or a.guess_scale<=0:raise ValueError('positive finite horizon guess scale required')
    if a.flow_iterations<1:raise ValueError('positive flow iteration count required')
    if not math.isfinite(a.domain_half_width) or a.domain_half_width<=0:raise ValueError('positive finite domain half width required')
    common_center=list(map(float,a.common_center.split(',')))
    if len(common_center)!=3 or not np.isfinite(common_center).all():raise ValueError('finite three-component common center required')
    if a.common:
        if a.common_radius is None or not math.isfinite(a.common_radius) or a.common_radius<=0:
            raise ValueError('positive finite common radius required')
        if max(abs(x) for x in common_center)+a.common_radius>=a.domain_half_width:
            raise ValueError('initial common sphere must fit inside the mesh domain')
    elif a.common_radius is not None or a.common_center!='0,0,0':
        raise ValueError('common center/radius options require --common')
    surface_count=1 if a.common else 2
    guess_files=[Path(x).resolve(strict=True) for x in a.initial_shapes] if a.initial_shapes else None
    if guess_files is not None and len(guess_files)!=surface_count:
        raise ValueError('one initial coefficient file per surface required')
    exe=Path(a.executable).resolve(strict=True);source=checkpoint_metadata(Path(a.checkpoint).resolve(strict=True))
    if any(max(abs(x) for x in hole[1:4])+radius>=a.domain_half_width for hole,radius in zip(source['holes'],source['inner_max'])):
        raise ValueError('mesh domain does not enclose the active holes and modified balls')
    if source['acceptance'] not in ('preliminary','strong') and not (a.allow_diagnostic and source['acceptance']=='diagnostic'):
        raise ValueError('checked binary or explicit --allow-diagnostic required')
    migration=validate_migration(a.migration_proof,source) if a.migration_proof else None
    levels=list(map(int,a.levels.split(',')))
    if len(levels)<3 or any(l<2 for l in levels) or any(x>=y for x,y in zip(levels,levels[1:])):
        raise ValueError('three increasing harmonic orders required')
    if a.enclosure_axis and levels[-1]>256:
        raise ValueError('axial enclosure certificates support harmonic orders through256')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    result=dict(executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source=source,
                initial_time=0,evolution_steps=0,geometry='direct_native',cpu_threads=a.geometry_threads,strict_expansion=a.strict_expansion,flow_alpha=a.flow_alpha,seed_horizon_guess_scale=a.guess_scale,flow_iterations=a.flow_iterations,domain_half_width=a.domain_half_width,
                criteria=dict(expansion_rms=1e-7,area_relative_spectral=1e-5,
                              shape_uniform_spectral=1e-4,area_relative_quadrature=1e-7),
                records=[],passed=False,horizon_enclosure_verified=False,
                stronger_binary_validation_complete=False)
    result['sampler_migration']=migration
    result['reuse_converged_shapes']=a.reuse_shapes
    result['initial_shape_guesses']=[str(x) for x in guess_files] if guess_files else []
    result['enclosure_method']='axial_interval' if a.enclosure_axis else 'monopole_and_Cauchy_tail'
    if a.enclosure_axis:
        import harmonic_enclosure
        helper=Path(harmonic_enclosure.__file__).resolve(strict=True)
        result['enclosure_implementation']=dict(path=str(helper),sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
        result['enclosure_axis']=a.enclosure_axis;result['enclosure_intervals']=a.enclosure_intervals
    result['surface_kind']='common' if a.common else 'component'
    result['common_search_initial_center']=common_center if a.common else None
    result['common_search_initial_radius']=a.common_radius if a.common else None
    input_template=(ROOT/'inputs/hispid.athinput').read_text()
    result['input_template_sha256']=hashlib.sha256(input_template.encode()).hexdigest()
    schedule=[(l,2*l,'spectral') for l in levels]+[(levels[-1],3*levels[-1],'quadrature')]
    if a.consumer_memory_mib<2048:raise ValueError('consumer screen needs at least2048MiB')
    estimates=[dict(lmax=l,ntheta=nt,harmonic_table_bytes=surface_count*harmonic_table_bytes(l,nt,a.harmonic_storage)) for l,nt,_ in schedule]
    if any(row['harmonic_table_bytes']+1024**3>a.consumer_memory_mib*1024**2 for row in estimates):
        raise ValueError('declared Serial harmonic storage plus allowance exceeds the consumer budget')
    result['harmonic_storage']=a.harmonic_storage
    result['consumer_memory_screen']=dict(budget_mib=a.consumer_memory_mib,other_allowance_bytes=1024**3,
        estimates=estimates,number_of_surfaces=surface_count,measured_peak=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(a.geometry_threads),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_PROC_BIND='spread',OMP_PLACES='cores')
    previous_shapes=None;previous_centers=None
    for lmax,ntheta,kind in schedule:
        if hashlib.sha256(exe.read_bytes()).hexdigest()!=result['executable_sha256']:
            raise ValueError('AthenaK executable changed before horizon worker')
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['file_sha256']:
            raise ValueError('checkpoint changed before horizon worker')
        if migration and validate_migration(a.migration_proof,source)!=migration:
            raise ValueError('sampler proof changed before horizon worker')
        run=root/f'{kind}_l{lmax}_n{ntheta}';run.mkdir()
        # AthenaK only permits command-line overrides of keys already present.
        extra=['use_puncture_massweighted_center_0 = false']
        extra += [f'center_{axis}_0 = {common_center[d]}' for d,axis in enumerate(('x','y','z'))]
        extra += [key+'_1 = '+value for key,value in (
            ('use_puncture','-1'),('start_time','0'),('stop_time','0'),
            ('flow_iterations',str(a.flow_iterations)),('flow_alpha_beta_const',str(a.flow_alpha)),
            ('expansion_rms_tol','1e-7'),('mass_tol','1e-12'),
            ('hmean_tol','100'),('use_puncture_massweighted_center','false'))]
        input_path=run/'binary.athinput'
        guess_inputs={};guess_counts=[]
        if guess_files:
            for h,path in enumerate(guess_files):
                coefficients=np.atleast_2d(np.loadtxt(path))
                count=coefficients.size
                if (coefficients.shape[0]!=1 or not np.isfinite(coefficients).all()
                    or math.isqrt(count)**2!=count or count>(lmax+1)**2):
                    raise ValueError('initial guess requires one complete supported finite SH record')
                guess_inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();guess_counts.append(count)
        shape_options=''.join(f'\nhispid_horizon_shape_guess_{h} = {path}' for h,path in enumerate(guess_files or []))
        input_path.write_text(input_template.replace(
            '<problem>','\n'.join(extra)+'\n<problem>'+shape_options+'\nhispid_parallel_geometry = '+str(a.geometry_threads>1).lower()))
        cmd=[str(exe),'-i',str(input_path),
             'problem/hispid_filename='+source['path'],'problem/hispid_source_sha256='+source['source_library_sha256'],
             f'problem/hispid_horizon_guess_scale={a.guess_scale}',
             'fastflow/factorized_harmonics='+str(a.harmonic_storage=='factorized').lower(),
             f'fastflow/num_horizons={surface_count}',f'fastflow/lmax={lmax}',f'fastflow/ntheta={ntheta}']
        if a.common:
            cmd += ['problem/hispid_common_horizon=true','problem/hispid_seed_horizon_guess=false',
                    f'fastflow/initial_radius_0={a.common_radius}']
        if a.allow_diagnostic:cmd += ['problem/hispid_allow_diagnostic=true']
        if migration:cmd += ['problem/hispid_allow_library_migration=true']
        for d in (1,2,3):cmd += [f'mesh/nx{d}=8',f'meshblock/nx{d}=8',f'mesh/x{d}min={-a.domain_half_width}',f'mesh/x{d}max={a.domain_half_width}']
        for h in range(surface_count):
            cmd += [f'fastflow/use_puncture_{h}=-1',f'fastflow/start_time_{h}=0',f'fastflow/stop_time_{h}=0',
                    f'fastflow/flow_iterations_{h}={a.flow_iterations}',f'fastflow/flow_alpha_beta_const_{h}={a.flow_alpha}',
                    f'fastflow/expansion_rms_tol_{h}='+('1e-5' if not a.strict_expansion and lmax<levels[-1] else '1e-7'),
                    f'fastflow/mass_tol_{h}=1e-12',f'fastflow/hmean_tol_{h}=100',
                    f'fastflow/use_puncture_massweighted_center_{h}=false']
        start=time.monotonic();row=dict(lmax=lmax,ntheta=ntheta,kind=kind,command=cmd,passed=False,holes=[])
        row['input_sha256']=hashlib.sha256(input_path.read_bytes()).hexdigest()
        row['input_path']=str(input_path.resolve());row['bound_inputs_unchanged']=False
        row['retained_artifacts_sha256']=guess_inputs.copy()
        row['initial_shape_inputs']=guess_inputs
        result['records'].append(row);(root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
        with (run/'run.log').open('w') as log:
            try:completed=subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
            except subprocess.TimeoutExpired:row['returncode']='timeout'
            else:row['returncode']=completed.returncode
        row['seconds']=time.monotonic()-start
        log_path=(run/'run.log').resolve();log_bytes=log_path.read_bytes()
        row['retained_artifacts_sha256'][str(log_path)]=hashlib.sha256(log_bytes).hexdigest()
        (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
        stdout=log_bytes.decode()
        row['shape_guess_verified']=all(
            f'HiSpID horizon_shape_guess horizon={h} coefficients={count}' in stdout
            for h,count in enumerate(guess_counts))
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in guess_inputs.items()):
            raise ValueError('initial horizon guess changed during worker')
        if hashlib.sha256(log_path.read_bytes()).hexdigest()!=row['retained_artifacts_sha256'][str(log_path)]:
            row['worker_log_changed']=True
            (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
            raise ValueError('worker log changed during decoding; retained attempt is unqualified')
        row['harmonic_storage']=a.harmonic_storage
        row['harmonic_allocation']=storage_evidence(stdout,a.harmonic_storage,lmax,ntheta,surface_count)
        (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
        if (hashlib.sha256(exe.read_bytes()).hexdigest()!=result['executable_sha256']
            or hashlib.sha256(input_path.read_bytes()).hexdigest()!=row['input_sha256']):
            row['executable_or_input_changed']=True
            (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
            raise ValueError('AthenaK executable/input changed during worker; retained attempt is unqualified')
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['file_sha256']:
            row['checkpoint_changed']=True
            (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
            raise ValueError('checkpoint changed during horizon worker; retained attempt is unqualified')
        if migration and validate_migration(a.migration_proof,source)!=migration:
            raise ValueError('sampler proof changed during horizon worker')
        row['bound_inputs_unchanged']=True
        witness=re.search(r'HiSpID horizon_geometry parallel=1 host_concurrency=(\d+)',stdout)
        row['parallel_geometry_verified']=bool(a.geometry_threads==1 or (witness and int(witness[1])==a.geometry_threads))
        row['import']=import_evidence(stdout,source,migration)
        row['zero_evolution_verified']=bool(re.search(r'time=0\.000000e\+00 cycle=0',stdout) and 'MeshBlock-cycles = 0' in stdout)
        row['attempts']=[]
        for match in re.finditer(r'HiSpID horizon (\d+) geometry=(\w+) found=([01]).*? attempt_area=(\S+) attempt_expansion_rms=(\S+)',stdout):
            area,rms=float(match[4]),float(match[5])
            row['attempts'].append(dict(index=int(match[1]),geometry=match[2],found=match[3]=='1',
                area=area if math.isfinite(area) else None,expansion_rms=rms if math.isfinite(rms) else None))
        if row['returncode']!=0:row['failure_log_tail']=stdout.splitlines()[-8:]
        shapes=[];centers=[]
        if row['returncode']==0:
            for h in range(surface_count):
                summary_path=run/f'hispid.horizon_summary_{h}.txt';shape_path=run/f'hispid.horizon_shape_{h}.txt'
                bound_surfaces={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (summary_path,shape_path)}
                row['retained_artifacts_sha256'].update(bound_surfaces)
                (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
                summary=np.atleast_2d(np.loadtxt(summary_path))[-1]
                shape=np.atleast_2d(np.loadtxt(shape_path))[-1];shapes.append(shape)
                if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in bound_surfaces.items()):
                    raise ValueError('horizon surface changed during decoding')
                match=re.search(r'HiSpID horizon '+str(h)+r' .*center_x=([\deE+.-]+) center_y=([\deE+.-]+) center_z=([\deE+.-]+)',stdout)
                if match is None:raise ValueError('actual finder center is required')
                center=np.array(list(map(float,match.groups())))
                centers.append(center)
                cauchy_lower=radius_lower_bound(shape);lower=cauchy_lower
                upper=float(2*shape[0]/math.sqrt(4*math.pi)-lower)
                certificate=None
                if a.enclosure_axis:
                    certificate=harmonic_enclosure.axial_range(shape,a.enclosure_axis,a.enclosure_intervals)
                    lower=certificate['radius_lower_bound'];upper=certificate['radius_upper_bound']
                offset=float(np.linalg.norm(center-np.array(source['holes'][h][1:4])))
                rounding_margin=float(64*np.finfo(float).eps*(1+uniform_bound(shape)+offset+source['inner_max'][h]))
                margin=lower-offset-source['inner_max'][h]-rounding_margin
                hole=dict(index=h,summary=summary.tolist(),coefficients=shape.tolist(),center=center.tolist(),
                          area=float(summary[7]),mass=float(summary[2]),coordinate_spin=summary[3:7].tolist(),
                          irreducible_mass=float(np.sqrt(summary[7]/(16*math.pi))),coordinate_spin_chi=float(summary[6]/summary[2]**2),
                          expansion_rms=float(np.sqrt(summary[8])),sampled_min_radius=float(summary[11]),
                          continuous_radius_lower_bound=lower,continuous_radius_upper_bound=upper,
                          center_offset=offset,rounding_allowance=rounding_margin,inner_ball_margin=margin)
                if certificate:hole.update(cauchy_radius_lower_bound=cauchy_lower,continuous_range_certificate=certificate)
                hole['expansion_pass']=bool(np.isfinite(summary).all() and summary[7]>0 and summary[8]>=0 and hole['expansion_rms']<1e-7)
                hole['retained_surface_encloses_inner_ball']=margin>0 and lower>0
                if a.common:
                    enclosed=[]
                    for index,(seed,radius) in enumerate(zip(source['holes'],source['inner_max'])):
                        distance=float(np.linalg.norm(center-np.array(seed[1:4])))
                        allowance=float(64*np.finfo(float).eps*(1+uniform_bound(shape)+distance+radius))
                        enclosed.append(dict(index=index,center_offset=distance,inner_radius=radius,
                            rounding_allowance=allowance,continuous_enclosure_margin=lower-distance-radius-allowance))
                    hole['component_ball_enclosures']=enclosed
                    hole['inner_ball_margin']=min(e['continuous_enclosure_margin'] for e in enclosed)
                    hole['retained_surface_encloses_inner_ball']=bool(lower>0 and hole['inner_ball_margin']>0)
                    hole['surface_kind']='common'
                if previous_shapes is not None:
                    hole['coefficient_change_uniform_bound']=shape_change(previous_shapes[h],shape)
                    hole['center_change']=float(np.linalg.norm(center-previous_centers[h]))
                    hole['shape_change_uniform_bound']=hole['coefficient_change_uniform_bound']+hole['center_change']
                row['holes'].append(hole)
            if not a.common:
                row['component_separation_margin']=float(np.linalg.norm(centers[0]-centers[1])-
                    sum(h['continuous_radius_upper_bound']+h['rounding_allowance'] for h in row['holes']))
                row['distinct_components_verified']=row['component_separation_margin']>0
            row['surface_kind_verified']=bool(not a.common or ' kind=common' in stdout)
            row['passed']=row['shape_guess_verified'] and row.get('import',{}).get('passed',False) and row['harmonic_allocation']['passed'] and row['zero_evolution_verified'] and row['surface_kind_verified'] and (a.common or row['distinct_components_verified']) and all(
                h['expansion_pass'] and h['retained_surface_encloses_inner_ball'] for h in row['holes'])
        (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(lmax,ntheta,kind,'returncode',row['returncode'],'passed',row['passed'],'seconds',row['seconds'],flush=True)
        for h in row['holes']:print({k:v for k,v in h.items() if k not in ('coefficients','summary')},flush=True)
        if row['returncode']!=0:break
        previous_shapes=shapes;previous_centers=centers
        guess_files=[(run/f'hispid.horizon_shape_{h}.txt').resolve() for h in range(surface_count)] if a.reuse_shapes and row['passed'] else None
    rows=result['records']
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=result['executable_sha256']:
        raise ValueError('AthenaK executable changed before final horizon qualification')
    if migration and validate_migration(a.migration_proof,source)!=migration:
        raise ValueError('sampler proof changed before final horizon qualification')
    if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['file_sha256']:
        raise ValueError('checkpoint changed before final horizon qualification')
    if (hashlib.sha256((ROOT/'inputs/hispid.athinput').read_bytes()).hexdigest()!=result['input_template_sha256']
        or any(hashlib.sha256(Path(r['input_path']).read_bytes()).hexdigest()!=r['input_sha256'] for r in rows)):
        raise ValueError('bound horizon inputs changed before final qualification')
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for r in rows for path,sha in r['retained_artifacts_sha256'].items()):
        raise ValueError('bound horizon logs or surfaces changed before final qualification')
    if a.enclosure_axis and hashlib.sha256(helper.read_bytes()).hexdigest()!=result['enclosure_implementation']['sha256']:
        raise ValueError('enclosure implementation changed before final qualification')
    result['schedule_prerequisites_verified']=schedule_prerequisites(rows,len(schedule))
    if result['schedule_prerequisites_verified']:
        fine,coarse,quad=rows[-2],rows[-3],rows[-1];checks=[]
        for h in range(surface_count):
            f,c,q=fine['holes'][h],coarse['holes'][h],quad['holes'][h]
            spectral_area=abs(f['area']/c['area']-1);quadrature_area=abs(q['area']/f['area']-1)
            mass_change=max(abs(f['mass']/c['mass']-1),abs(q['mass']/f['mass']-1))
            spins=[np.array(v['coordinate_spin'][:3])/v['mass']**2 for v in (c,f,q)]
            spin_change=max(float(np.linalg.norm(spins[1]-spins[0])),float(np.linalg.norm(spins[2]-spins[1])))
            buffer=2*(f['shape_change_uniform_bound']+q['shape_change_uniform_bound'])
            buffer_rounding=(64*(lmax+1)*np.finfo(float).eps*(1+buffer+uniform_bound(f['coefficients'])+uniform_bound(q['coefficients'])) if a.enclosure_axis else 0.)
            check=dict(index=h,mass_relative_change=mass_change,dimensionless_spin_vector_change=spin_change,spectral_area_relative_change=spectral_area,quadrature_area_relative_change=quadrature_area,
                       observed_refinement_buffer=buffer,buffer_rounding_allowance=buffer_rounding,enclosure_margin_after_buffer=q['inner_ball_margin']-buffer-buffer_rounding)
            check['passed']=bool(fine['passed'] and quad['passed'] and spectral_area<1e-5 and quadrature_area<1e-7
                                 and mass_change<1e-4 and spin_change<1e-4 and f['shape_change_uniform_bound']<1e-4 and check['enclosure_margin_after_buffer']>0)
            checks.append(check)
        result['refinement_checks']=checks;result['passed']=all(c['passed'] for c in checks) and (not a.strict_expansion or all(row['passed'] for row in rows))
        result['horizon_enclosure_verified']=result['passed']
        result['common_horizon_verified']=bool(a.common and result['passed'])
    result['note']='Continuous bounds apply to the retained harmonic surfaces. The observed refinement buffer is an empirical truncation check, not a rigorous bound on the exact PDE surface. g/operator modified balls alone are tested; noncompact f/F attenuation tails require exterior constraints. Reported spin is the coordinate rotation integral, not an approximate-Killing-vector spin. Input constraint acceptance remains separate; diagnostic or preliminary input is not promoted to strong binary validation.'
    (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
