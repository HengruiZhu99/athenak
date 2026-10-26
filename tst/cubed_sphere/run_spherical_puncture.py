#!/usr/bin/env python3
"""Native-gauge wormhole controls on a spherical atlas, with a Minkowski target."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
from run_conformal_spherical import run_case,sha


def parameters(steps=0,factor=1):
    # Match the original N24 puncture collar, not the wave-map pulse's collar.
    L=.25245594360687235
    p={f'mesh/x{a}{end}':sign*L for a in [1,2,3] for end,sign in [('min',-1),('max',1)]}
    p.update({'cubed_sphere/conformal_mass':.5,'cubed_sphere/conformal_curvature_radius':1,
        'cubed_sphere/conformal_layer_r0':.35,'cubed_sphere/conformal_layer_r1':.75,
        'cubed_sphere/conformal_native_layer_gauge':True,
        'cubed_sphere/conformal_fixed_dt':.00028515624999999044*factor,
        'cubed_sphere/conformal_probe_steps':steps})
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('binary',type=Path);ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--mode',choices=['admission','evolution','refinement','radial'],default='admission')
    ap.add_argument('--timeout',type=int,default=450);ap.add_argument('--tlim',type=float,default=.2)
    ap.add_argument('--cpu-golden',type=Path)
    args=ap.parse_args();assert 0<args.timeout<=450
    source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'athena_frozen';shutil.copy2(args.binary.resolve(),binary)
    files=[*source.glob('src/**/*.cpp'),*source.glob('src/**/*.hpp'),source/'CMakeLists.txt',source/'src/CMakeLists.txt',
        source/'inputs/z4c/cubed_sphere_conformal_probe.athinput',Path(__file__).resolve(),Path(__file__).with_name('run_conformal_spherical.py')]
    record={'mode':args.mode,'binary_sha256':sha(binary),
        'source_sha256':{str(p.relative_to(source)):sha(p) for p in sorted(set(files)) if not p.name.startswith('._')},
        'scope':'Same M=.5 LayerWormhole initial data and native physical-trace layer gauge; fixed initial reconstruction, independent Minkowski reference throughout. One rank, uniform bulk. Different grid, mapped derivatives and outer ghosts.'}
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    if args.mode=='admission':
        configurations=[('two-step','0.0',.001,3,2,1,16,12,41),('t002','0.1',.02,3,0,1,16,12,41)]
    elif args.mode=='evolution':
        configurations=[('degree3','0.0',args.tlim,3,0,1,16,12,41),('degree4','0.1',args.tlim,4,0,1,16,12,41),
                        ('degree3-halfstep','1.0',args.tlim,3,0,.5,16,12,41)]
    elif args.mode=='refinement':
        configurations=[(f'B{bulk}-A{angular}',f'{tile}.0',args.tlim,3,0,1,bulk,angular,41)
                        for tile,(bulk,angular) in enumerate([(16,12),(32,12),(16,24),(32,24)])]
    else:
        configurations=[(f'R{nr}',f'{tile}.0',args.tlim,3,0,1,16,12,nr)
                        for tile,nr in enumerate([41,82,123])]
    def launch(config):
        name,tile,tlim,degree,steps,factor,bulk,angular,nr=config
        extra=parameters(steps,factor)
        if args.mode=='refinement':
            # Common cap lies below both angular grids' pole CFL. The unchanged
            # map/domain is sampled differently; minimum active Omega may vary.
            extra['cubed_sphere/conformal_fixed_dt']=.00025
        if args.mode=='radial': extra['cubed_sphere/conformal_fixed_dt']=.00008
        return run_case(binary,source,out,name,tile,bulk,nr,angular,0,5,.15*factor,
                        tlim,degree,args.timeout,.04*factor,extra)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={c[0]:pool.submit(launch,c) for c in configurations}
        cases={name:job.result() for name,job in jobs.items()}
    record['cases']=cases
    if args.mode=='admission':
        a=cases['two-step']
        assert a['reached_step_target'] and a['reference_unchanged'] and a['initial_profile_unchanged'] and a['all_rows_finite']
        initial=list(csv.DictReader((out/'two-step/cubed_sphere_conformal.csv').open()))[:2]
        assert max(float(r['H_linf']) for r in initial)<1e-9
        if args.cpu_golden:
            b=json.loads(args.cpu_golden.read_text())['last_rows']
            keys=['H_l2','H_linf','M_conformal_l2','Z_conformal_l2','Theta_l2','min_alpha','min_chi','max_deviation']
            differences={x['region']:{k:abs(float(x[k])-float(y[k]))/max(abs(float(y[k])),1e-20) for k in keys}
                         for x,y in zip(a['last_rows'],b)}
            assert max(v for r in differences.values() for v in r.values())<1e-7,differences
            record['CPU_GPU_relative_difference']=differences
        record['admission_pass']=True
    (out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
