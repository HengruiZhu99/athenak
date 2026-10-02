"""Initial-time AthenaK ADM constraint refinement for exact seed imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cases',default='schwarzschild,kerr95,boost885,kerr95_boost885')
    p.add_argument('--levels',default='16,32,64');a=p.parse_args()
    levels=list(map(int,a.levels.split(',')))
    if len(levels)<3 or any(n%2 or n<16 for n in levels) or any(x>=y for x,y in zip(levels,levels[1:])):
        raise ValueError('three or more increasing even mesh resolutions are required')
    exe=Path(a.executable).resolve(strict=True);manifest=json.loads(Path(a.manifest).read_text())
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    result=dict(executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),initial_time=0,
                criteria=dict(outer_constraint_rms=1e-6,outer_constraint_max=1e-4,
                              absolute_roundoff_floor=1e-12),
                common_region='cube [-2,2]^3 outside radius1.8 from active puncture',
                records=[],passed=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for case in a.cases.split(','):
        source=manifest[case];rows=[]
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['file_sha256']:
            raise ValueError('checkpoint file differs from manifest')
        for n in levels:
            run=root/f'{case}_n{n}';run.mkdir(exist_ok=False)
            cmd=[str(exe),'-i',str(ROOT/'inputs/hispid.athinput'),
                 'problem/hispid_filename='+source['path'],
                 'problem/hispid_source_sha256='+source['source_library_sha256'],
                 'problem/hispid_initial_horizons=false','problem/hispid_mesh_constraints=true',
                 'problem/hispid_mesh_constraint_min_radius=1.8']
            for d in (1,2,3):cmd += [f'mesh/nx{d}={n}',f'meshblock/nx{d}={n}']
            start=time.monotonic()
            completed=subprocess.run(cmd,cwd=run,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
            (run/'run.log').write_text(completed.stdout)
            row=dict(case=case,resolution=n,command=cmd,source=source,returncode=completed.returncode,
                     seconds=time.monotonic()-start,passed_local=False)
            row['zero_evolution_verified']=bool(re.search(r'time=0\.000000e\+00 cycle=0',completed.stdout) and 'MeshBlock-cycles = 0' in completed.stdout)
            if completed.returncode==0:
                row['mesh']=json.loads((run/'hispid.hispid_mesh_constraints.json').read_text())
                outer=row['mesh']['regions'][2]
                row['passed_local']=bool(outer['volume']>0 and row['zero_evolution_verified']
                    and row['mesh']['roundtrip_relative_error']<1e-11
                    and row['mesh']['consumer_library_sha256']==source['source_library_sha256']
                    and all(outer[k]<1e-6 for k in ('H_rms','M_rms'))
                    and all(outer[k]<1e-4 for k in ('H_max','M_max')))
                print(case,n,outer,flush=True)
            else:print(completed.stdout[-2500:],flush=True)
            result['records'].append(row);rows.append(row)
            (root/'mesh.json').write_text(json.dumps(result,indent=2)+'\n')
        for row in rows:row['passed_case']=False
        if all('mesh' in row for row in rows):
            converges=all(rows[i+1]['mesh']['regions'][2][k]<rows[i]['mesh']['regions'][2][k]
                          or max(rows[i+1]['mesh']['regions'][2][k],rows[i]['mesh']['regions'][2][k])<1e-12
                          for i in range(len(rows)-1) for k in ('H_rms','M_rms'))
            rows[-1]['passed_case']=bool(rows[-1]['passed_local'] and converges)
            rows[-1]['converges_or_exact_zero']=bool(converges)
    result['passed']=all([row for row in result['records'] if row['case']==case][-1]['passed_case']
                         for case in a.cases.split(','))
    (root/'mesh.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
