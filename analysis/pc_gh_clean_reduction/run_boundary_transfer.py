#!/usr/bin/env python3
"""Run the frozen small physical-boundary operator coverage cases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--ranks',type=int,default=1)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
recipes=[('outflow',True,2),('reflect',False,2),('reflect',True,2),
         ('mixed',False,2),('mixed',True,2),('outflow',False,3),
         ('outflow',True,3),('outflow',False,4),('outflow',True,4)]
records=[]
for boundary,smr,extrap in recipes:
    name=f'{boundary}-{"smr" if smr else "uniform"}-e{extrap}'
    folder=args.output/name
    command=[sys.executable,str(Path(__file__).with_name('run_transfer_mesh.py').resolve()),
             '--binary',str(args.binary.resolve()),'--output',str(folder.resolve()),
             '--boundary',boundary,'--extrap-order',str(extrap),'--ranks',str(args.ranks)]
    if smr:command+=['--smr']
    start=time.time()
    with (args.output/f'{name}.log').open('w') as log:
        run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    values=json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else []
    result=dict(name=name,command=command,returncode=run.returncode,
                wall_seconds=time.time()-start,
                status='PASS' if len(values)==6 and all(r['status']=='PASS' for r in values) else 'FAIL')
    records.append(result)
    (args.output/'controller.json').write_text(json.dumps(records,indent=2)+'\n')
    print(name,result['status'],flush=True)
