#!/usr/bin/env python3
"""Continue the frozen baseline alongside fresh half-CFL temporal controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from run_conformal_spherical import run_case,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--mode',choices=['evolution'],default='evolution')
    p.add_argument('--affinity',choices=['root','tile'],default='tile')
    p.add_argument('--wall',type=float,default=300);p.add_argument('--tlim',type=float,default=2)
    a=p.parse_args();assert 0<a.wall<=360 and a.tlim==2
    source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'athena_frozen';shutil.copy2(a.binary.resolve(),binary);tag=sha(binary)
    previous=json.loads((a.previous/'summary.json').read_text());assert previous['binary_sha256']==tag
    files=[*source.glob('src/**/*.cpp'),*source.glob('src/**/*.hpp'),source/'CMakeLists.txt',source/'src/CMakeLists.txt',
           source/'inputs/z4c/cubed_sphere_conformal_probe.athinput',Path(__file__).resolve(),
           Path(__file__).with_name('run_conformal_spherical.py')]
    record={'mode':'evolution','binary_sha256':tag,'affinity':a.affinity,'previous':str(a.previous),
            'source_sha256':{str(f.relative_to(source)):sha(f) for f in sorted(set(files)) if not f.name.startswith('._')},
            'scope':'B16/A12/R48 matched collar; original physical parameters, degrees 2/4, each full-CFL or half-CFL. Numerical source/binary unchanged. Independent cases share one node, not state.'}
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    def run(name,degree,index,half,old):
        extra={'cubed_sphere/radial_stretch':2.8592697712828787,
               'cubed_sphere/conformal_probe_steps':0,'cubed_sphere/conformal_fixed_dt':0,
               'cubed_sphere/conformal_checkpoint':str(out/name/'probe_checkpoint.bin'),
               'cubed_sphere/conformal_checkpoint_tag':tag,'cubed_sphere/conformal_wall_seconds':a.wall,
               'cubed_sphere/conformal_max_cycles':100000}
        if old:
            cp=old['checkpoint_payload'];assert sha(Path(cp['path']))==cp['sha256']
            extra['cubed_sphere/conformal_restart']=cp['path']
        tile=str(index) if a.affinity=='root' else f'{index}.0'
        return run_case(binary,source,out,name,tile,16,48,12,.2,10,.05 if half else .1,
                        a.tlim,degree,420,.015 if half else .03,extra)
    cases={};jobs={}
    configs=[('degree2',2,False),('degree4',4,False),('half-degree2',2,True),('half-degree4',4,True)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index,(name,degree,half) in enumerate(configs):
            old=previous['cases'].get(name)
            if old and old['outcome']!='checkpointed':
                cases[name]={**old,'continued_from_previous_terminal_case':str(a.previous)};continue
            jobs[name]=pool.submit(run,name,degree,index,half,old)
        cases.update({name:job.result() for name,job in jobs.items()})
    record['cases']=cases
    record['all_terminal']=all(r['outcome'] in ['complete','failed','timeout'] for r in cases.values())
    (out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'binary_sha256':tag,'all_terminal':record['all_terminal']}))


if __name__=='__main__':main()
