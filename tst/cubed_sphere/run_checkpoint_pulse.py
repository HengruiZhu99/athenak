#!/usr/bin/env python3
"""Matched-collar pulse, continued through short jobs with immutable references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from run_conformal_spherical import run_case,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=['admission','evolution'],required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--cpu-golden',type=Path)
    p.add_argument('--affinity',choices=['root','tile'],default='root')
    p.add_argument('--tlim',type=float,default=2);p.add_argument('--wall',type=float,default=300)
    a=p.parse_args();assert 0<a.wall<=360
    source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'athena_frozen';shutil.copy2(a.binary.resolve(),binary);tag=sha(binary)
    files=[*source.glob('src/**/*.cpp'),*source.glob('src/**/*.hpp'),source/'CMakeLists.txt',source/'src/CMakeLists.txt',
           source/'inputs/z4c/cubed_sphere_conformal_probe.athinput',Path(__file__).resolve(),Path(__file__).with_name('run_conformal_spherical.py')]
    record={'mode':a.mode,'binary_sha256':tag,'affinity':a.affinity,
            'source_sha256':{str(f.relative_to(source)):sha(f) for f in sorted(set(files)) if not f.name.startswith('._')},
            'scope':'Original amplitude .2, kappa1=10, a=.5 and layer [.05,.95], pole CFL .03; B16/A12/R48 exponential map matches archived N24 minimum Omega. Research state checkpoints, independent Minkowski reference throughout.'}
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    def run(name,degree,tile,steps=0,restart=None,wall=0,tlim=None,dt=0):
        extra={'cubed_sphere/radial_stretch':2.8592697712828787,
               'cubed_sphere/conformal_probe_steps':steps,'cubed_sphere/conformal_fixed_dt':dt,
               'cubed_sphere/conformal_checkpoint':str(out/name/'probe_checkpoint.bin'),
               'cubed_sphere/conformal_checkpoint_tag':tag,'cubed_sphere/conformal_wall_seconds':wall}
        if restart:
            assert sha(Path(restart['path']))==restart['sha256'],'changed previous checkpoint'
            extra['cubed_sphere/conformal_restart']=restart['path']
        return run_case(binary,source,out,name,tile,16,48,12,.2,10,.1,a.tlim if tlim is None else tlim,degree,420,.03,extra)
    if a.mode=='admission':
        full=run('full',2,'0',4,tlim=.001,dt=.00005)
        part=run('part',2,'0',2,tlim=.001,dt=.00005)
        resumed=run('resumed',2,'0',4,restart=part['checkpoint_payload'],tlim=.001,dt=.00005)
        assert full['outcome']==part['outcome']==resumed['outcome']=='complete'
        assert full['checkpoint_payload']['sha256']==resumed['checkpoint_payload']['sha256']
        assert all(r['reference_unchanged'] for r in [full,part,resumed])
        record['split_state_bitwise_equal']=True
        if a.cpu_golden:
            cpu=json.loads(a.cpu_golden.read_text())['last_rows'];diffs={}
            for x,y in zip(full['last_rows'],cpu):
                assert x['region']==y['region']
                diffs[x['region']]={k:abs(float(x[k])-float(y[k])) for k in ['H_l2','H_linf','M_conformal_l2','Z_conformal_l2','Theta_l2','min_alpha','min_chi','max_deviation']}
                assert all(diffs[x['region']][k]<=1e-12+1e-7*abs(float(y[k])) for k in diffs[x['region']]),diffs
            record['CPU_GPU_absolute_differences']=diffs
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs={label:pool.submit(run,label,2,device,tlim=.02,dt=.00005)
                  for label,device in [('root','0'),('tile','1.0')]}
            perf={name:j.result() for name,j in jobs.items()}
        assert all(r['outcome']=='complete' and r['reference_unchanged'] for r in perf.values())
        diffs={x['region']:{k:abs(float(x[k])-float(y[k]))/max(abs(float(y[k])),1e-20)
                           for k in ['H_l2','H_linf','M_conformal_l2','Z_conformal_l2','Theta_l2','min_alpha','min_chi','max_deviation']}
               for x,y in zip(perf['root']['last_rows'],perf['tile']['last_rows'])}
        assert max(v for r in diffs.values() for v in r.values())<1e-7,diffs
        record.update({'cases':{'full':full,'part':part,'resumed':resumed,**perf},
                       'root_tile_relative_differences':diffs,'admission_pass':True})
    else:
        previous=json.loads((a.previous/'summary.json').read_text()) if a.previous else None
        if previous:assert previous['binary_sha256']==tag
        cases={};jobs={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for index,degree in enumerate([2,4]):
                name=f'degree{degree}';old=previous['cases'].get(name) if previous else None
                if old and old['outcome']!='checkpointed':
                    cases[name]={**old,'continued_from_previous_terminal_case':str(a.previous)};continue
                tile=str(index) if a.affinity=='root' else f'{index}.0'
                jobs[name]=pool.submit(run,name,degree,tile,restart=old['checkpoint_payload'] if old else None,wall=a.wall)
            cases.update({name:j.result() for name,j in jobs.items()})
        record['cases']=cases;record['previous']=str(a.previous) if a.previous else None
        record['all_terminal']=all(r['outcome'] in ['complete','failed','timeout'] for r in cases.values())
    (out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:record[k] for k in ['mode','binary_sha256','admission_pass','all_terminal'] if k in record}))


if __name__=='__main__':main()
