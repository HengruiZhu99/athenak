#!/usr/bin/env python3
"""Two-step SERIAL/MPI-one references for GPU smoke comparison, never long runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
import control
import native_transition_campaign as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=list(c.VARIANTS),default=['A','B','C','D'])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binding=json.loads((HERE/'checks/native-matrix-build.json').read_text())
    require_source=c.SOURCE_COMMIT
    source=subprocess.check_output(['git','rev-parse',binding['scientific_source_commit']],cwd=ROOT,text=True).strip()
    assert source==require_source
    records=[]
    for v in a.variants:
        suffix=c.VARIANTS[v]['suffix'];record=next(t for t in binding['targets'] if t['variant']==suffix)
        exe=ROOT/record['binary'];assert control.digest(exe)==record['sha256']
        for kind in ['minkowski','puncture']:
            prefix=out/(v.lower()+'-'+kind)
            cmd=['mpiexec','-n','1',str(exe),'16',str(prefix),kind,'.001','.0005','-','-','600','.05','.001','12','24','2']
            with Path(str(prefix)+'.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads(Path(str(prefix)+'.report.json').read_text())
            for check in c.required_checks(v,'smoke',kind):
                if check['path'] in [['backend'],['mpi_size']]:continue
                assert control.evaluate_check(report,check)['passed'],check
            assert report['backend']=='Serial' and report['mpi_size']==1
            cp=Path(str(prefix)+'.checkpoint');verification=c.inspect_checkpoint(cp,v,control.digest(cp),kind,.001)
            records.append(dict(variant=v,kind=kind,command=cmd,binary_sha256=record['sha256'],verification=verification,
                artifacts={str(path.relative_to(out)):dict(bytes=path.stat().st_size,sha256=control.digest(path)) for path in out.glob(prefix.name+'.*')}))
            control.atomic(out/'results.json',dict(scope='Two-step Serial, one MPI rank; not GPU or long stability evidence',source_commit=require_source,cases=records))
            print(v,kind,'passed',flush=True)


if __name__=='__main__':main()
