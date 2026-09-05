"""Run one explicitly selected hybrid stage; numerical exits are not physics passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',required=True)
    ap.add_argument('--pattern',default='*.athinput')
    ap.add_argument('--ranks',type=int,default=1)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=sorted((args.inputs/args.stage).glob(args.pattern))
    if not cases: raise SystemExit('No selected inputs')
    report=[]
    for path in cases:
        destination=args.output/path.stem
        if destination.exists():
            raise SystemExit(f'Refusing implicit rerun/overwrite: {destination}')
        argv=[sys.executable,str(HERE/'cuda_driver.py'),'run','--build',str(args.build),
              '--input',str(path),'--output',str(destination),'--wall-segment','00:15:00',
              '--ranks',str(args.ranks)]
        status=subprocess.run(argv).returncode
        item=dict(input=str(path),run=str(destination),exit_code=status,
                  completed=datetime.now(timezone.utc).isoformat())
        if args.stage in ['oracle','mpi-oracle'] and status==0:
            with (destination/'independent-verification.log').open('x') as log:
                item['oracle_exit']=subprocess.run([sys.executable,str(HERE/'verify_hybrid_projection.py'),
                    str(destination),'--output',str(destination/'independent-verification.json')],
                    stdout=log,stderr=subprocess.STDOUT).returncode
        report.append(item)
        (args.output/'stage-results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(item,flush=True)
        if args.stage in ['oracle','mpi-oracle','flat'] and (status or item.get('oracle_exit')):
            raise SystemExit('Correctness stage failed; inspect preserved evidence')


if __name__=='__main__':main()
