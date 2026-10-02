"""Initial-time solved-binary FastFlow refinement and inner-ball enclosure.

Uses direct native geometry. Mesh round trips are checked by the pgen; this
is not a mesh-resolved horizon or production evolution test.
"""
import argparse,hashlib,json,math,os,re,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


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


def checkpoint_metadata(path):
    meta={}
    with path.open() as f:
        for line in f:
            words=line.split()
            if not words:continue
            if words[0]=='unknowns':break
            meta[words[0]]=words[1:]
    if meta.get('HISPID_CHECKPOINT')!=['1']:raise ValueError('checkpoint version')
    holes=[list(map(float,meta['hole'+str(h)])) for h in range(2)]
    if any(x[0]<=0 for x in holes):raise ValueError('two active holes required')
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_library_sha256=meta['library_sha256'][0],acceptance=meta['acceptance'][0],
                holes=holes,inner_max=list(map(float,meta['inner_max'])),
                inner_flatten=int(meta['inner_flatten'][0]),parameterization=meta['parameterization'][0])


def main():
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--levels',default='8,12,16')
    p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    exe=Path(a.executable).resolve(strict=True);source=checkpoint_metadata(Path(a.checkpoint).resolve(strict=True))
    if source['acceptance'] not in ('preliminary','strong'):raise ValueError('checked binary required')
    levels=list(map(int,a.levels.split(',')))
    if len(levels)<3 or any(l<2 for l in levels) or any(x>=y for x,y in zip(levels,levels[1:])):
        raise ValueError('three increasing harmonic orders required')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    result=dict(executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source=source,
                initial_time=0,evolution_steps=0,geometry='direct_native',cpu_threads=1,
                criteria=dict(expansion_rms=1e-7,area_relative_spectral=1e-5,
                              shape_uniform_spectral=1e-4,area_relative_quadrature=1e-7),
                records=[],passed=False,horizon_enclosure_verified=False,
                stronger_binary_validation_complete=False)
    schedule=[(l,2*l,'spectral') for l in levels]+[(levels[-1],3*levels[-1],'quadrature')]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    previous_shapes=None;previous_centers=None
    for lmax,ntheta,kind in schedule:
        run=root/f'{kind}_l{lmax}_n{ntheta}';run.mkdir()
        # AthenaK only permits command-line overrides of keys already present.
        extra=['use_puncture_massweighted_center_0 = false']
        extra += [key+'_1 = '+value for key,value in (
            ('use_puncture','-1'),('start_time','0'),('stop_time','0'),
            ('flow_iterations','600'),('flow_alpha_beta_const','1'),
            ('expansion_rms_tol','1e-7'),('mass_tol','1e-12'),
            ('hmean_tol','100'),('use_puncture_massweighted_center','false'))]
        input_path=run/'binary.athinput'
        input_path.write_text((ROOT/'inputs/hispid.athinput').read_text().replace(
            '<problem>','\n'.join(extra)+'\n<problem>'))
        cmd=[str(exe),'-i',str(input_path),
             'problem/hispid_filename='+source['path'],'problem/hispid_source_sha256='+source['source_library_sha256'],
             'fastflow/num_horizons=2',f'fastflow/lmax={lmax}',f'fastflow/ntheta={ntheta}']
        for d in (1,2,3):cmd += [f'mesh/nx{d}=8',f'meshblock/nx{d}=8',f'mesh/x{d}min=-8',f'mesh/x{d}max=8']
        for h in range(2):
            cmd += [f'fastflow/use_puncture_{h}=-1',f'fastflow/start_time_{h}=0',f'fastflow/stop_time_{h}=0',
                    f'fastflow/flow_iterations_{h}=600',f'fastflow/flow_alpha_beta_const_{h}=1',
                    f'fastflow/expansion_rms_tol_{h}='+('1e-5' if lmax<levels[-1] else '1e-7'),
                    f'fastflow/mass_tol_{h}=1e-12',f'fastflow/hmean_tol_{h}=100',
                    f'fastflow/use_puncture_massweighted_center_{h}=false']
        start=time.monotonic();row=dict(lmax=lmax,ntheta=ntheta,kind=kind,command=cmd,passed=False,holes=[])
        with (run/'run.log').open('w') as log:
            try:completed=subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
            except subprocess.TimeoutExpired:row['returncode']='timeout'
            else:row['returncode']=completed.returncode
        row['seconds']=time.monotonic()-start;stdout=(run/'run.log').read_text()
        row['zero_evolution_verified']=bool(re.search(r'time=0\.000000e\+00 cycle=0',stdout) and 'MeshBlock-cycles = 0' in stdout)
        shapes=[];centers=[]
        if row['returncode']==0:
            for h in range(2):
                summary=np.atleast_2d(np.loadtxt(run/f'hispid.horizon_summary_{h}.txt'))[-1]
                shape=np.atleast_2d(np.loadtxt(run/f'hispid.horizon_shape_{h}.txt'))[-1];shapes.append(shape)
                match=re.search(r'HiSpID horizon '+str(h)+r' .*center_x=([\deE+.-]+) center_y=([\deE+.-]+) center_z=([\deE+.-]+)',stdout)
                if match is None:raise ValueError('actual finder center is required')
                center=np.array(list(map(float,match.groups())))
                centers.append(center)
                lower=radius_lower_bound(shape);offset=float(np.linalg.norm(center-np.array(source['holes'][h][1:4])))
                upper=float(2*shape[0]/math.sqrt(4*math.pi)-lower)
                rounding_margin=float(64*np.finfo(float).eps*(1+uniform_bound(shape)+offset+source['inner_max'][h]))
                margin=lower-offset-source['inner_max'][h]-rounding_margin
                hole=dict(index=h,summary=summary.tolist(),coefficients=shape.tolist(),center=center.tolist(),
                          area=float(summary[7]),mass=float(summary[2]),coordinate_spin=summary[3:7].tolist(),
                          expansion_rms=float(np.sqrt(summary[8])),sampled_min_radius=float(summary[11]),
                          continuous_radius_lower_bound=lower,continuous_radius_upper_bound=upper,
                          center_offset=offset,rounding_allowance=rounding_margin,inner_ball_margin=margin)
                hole['expansion_pass']=bool(np.isfinite(summary).all() and summary[7]>0 and summary[8]>=0 and hole['expansion_rms']<1e-7)
                hole['retained_surface_encloses_inner_ball']=margin>0 and lower>0
                if previous_shapes is not None:
                    hole['coefficient_change_uniform_bound']=shape_change(previous_shapes[h],shape)
                    hole['center_change']=float(np.linalg.norm(center-previous_centers[h]))
                    hole['shape_change_uniform_bound']=hole['coefficient_change_uniform_bound']+hole['center_change']
                row['holes'].append(hole)
            row['component_separation_margin']=float(np.linalg.norm(centers[0]-centers[1])-
                sum(h['continuous_radius_upper_bound']+h['rounding_allowance'] for h in row['holes']))
            row['distinct_components_verified']=row['component_separation_margin']>0
            row['passed']=row['zero_evolution_verified'] and row['distinct_components_verified'] and all(
                h['expansion_pass'] and h['retained_surface_encloses_inner_ball'] for h in row['holes'])
        result['records'].append(row);(root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(lmax,ntheta,kind,'returncode',row['returncode'],'passed',row['passed'],'seconds',row['seconds'],flush=True)
        for h in row['holes']:print({k:v for k,v in h.items() if k not in ('coefficients','summary')},flush=True)
        if row['returncode']!=0:break
        previous_shapes=shapes;previous_centers=centers
    rows=result['records']
    if len(rows)==len(schedule) and all(r['returncode']==0 for r in rows):
        fine,coarse,quad=rows[-2],rows[-3],rows[-1];checks=[]
        for h in range(2):
            f,c,q=fine['holes'][h],coarse['holes'][h],quad['holes'][h]
            spectral_area=abs(f['area']/c['area']-1);quadrature_area=abs(q['area']/f['area']-1)
            buffer=2*(f['shape_change_uniform_bound']+q['shape_change_uniform_bound'])
            check=dict(index=h,spectral_area_relative_change=spectral_area,quadrature_area_relative_change=quadrature_area,
                       observed_refinement_buffer=buffer,enclosure_margin_after_buffer=q['inner_ball_margin']-buffer)
            check['passed']=bool(fine['passed'] and quad['passed'] and spectral_area<1e-5 and quadrature_area<1e-7
                                 and f['shape_change_uniform_bound']<1e-4 and check['enclosure_margin_after_buffer']>0)
            checks.append(check)
        result['refinement_checks']=checks;result['passed']=all(c['passed'] for c in checks)
        result['horizon_enclosure_verified']=result['passed']
    result['note']='Continuous bounds apply to the retained harmonic surfaces. The observed refinement buffer is an empirical truncation check, not a rigorous bound on the exact PDE surface. g/operator modified balls alone are tested; noncompact f/F attenuation tails require exterior constraints. Reported spin is the coordinate rotation integral, not an approximate-Killing-vector spin. Preliminary input is not promoted to strong binary validation.'
    (root/'binary.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
