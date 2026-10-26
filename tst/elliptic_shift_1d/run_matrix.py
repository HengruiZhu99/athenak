#!/usr/bin/env python3
"""Execute an explicit input matrix sequentially, preserving every outcome."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
data=json.loads(a.input.read_text());(a.output/'input.json').write_text(a.input.read_text());receipts=[]
for case in data['cases']:
    options={**data['common'],**case['arguments']}
    command=[sys.executable,str(Path(__file__).with_name('evolve.py')),
        '--library',a.library,'--fast-library',a.fast_library,'--output',str(a.output/case['name'])]
    for key,value in options.items():
        if isinstance(value,bool):
            if value:command+=['--'+key.replace('_','-')]
        elif value is not None:command+=['--'+key.replace('_','-'),str(value)]
    with (a.output/(case['name']+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipts.append(dict(name=case['name'],command=command,returncode=result.returncode))
    (a.output/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(receipts[-1]),flush=True)
