#!/usr/bin/env python3
"""Small initial AF finite-grid spectra for separated source controls."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
for name,family,profile,scale in [('linear','linear','physical-log',.1),('local-rational','rational','physical-local',.001)]:
 for n in [33,65,129]:
  for epsilon in [1e-6,1e-7]:
   output=a.output/f'{name}-n{n}-eps{epsilon}'
   command=[sys.executable,str(Path(__file__).with_name('diagnose_puncture.py')),'--library',a.library,'--fast-library',a.fast_library,'--output',str(output),'--n',str(n),'--mass',str(1/3),'--epsilon',str(epsilon),'--slicing-family',family,'--slicing-scale',str(scale),'--lapse-restoring-profile',profile,'--lapse-background-source','collapsed-inner']
   with output.with_suffix('.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
   assert r.returncode==0,(name,n,epsilon,r.returncode)
   record=json.loads((output/'review.json').read_text());records.append(record)
   (a.output/'review.json').write_text(json.dumps(dict(records=records,scope='Initial nonstationary finite-grid Jacobians; no standalone continuum shock/stability theorem'),indent=2)+'\n')
   print(json.dumps(dict(name=name,n=n,epsilon=epsilon,dominant=record['modes'][0])),flush=True)
