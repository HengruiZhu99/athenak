#!/usr/bin/env python3
"""Source-bound executable AF evolution comparison through the steep-front phase."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=129);p.add_argument('--t',type=float,default=6);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
common=[sys.executable,str(Path(__file__).with_name('evolve.py')),'--library',a.library,'--backend','spherical-g1','--n',str(a.n),'--puncture','--puncture-chart','--initial-data','wormhole','--mass',str(1/3),'--lapse-restoring','1','--lapse-restoring-form','multiplicative','--lapse-restoring-profile','physical-log','--lapse-background-source','collapsed-inner','--advection','upwind','--scri-sat','4','--regularity-sat','8','--puncture-ko','.5','--report-dt','.1','--t',str(a.t),'--cfl','.05']
for name in ['python','compiled']:
 command=common+['--output',str(a.output/name)]
 if name=='compiled':command+=['--fast-library',a.fast_library]
 with (a.output/(name+'.log')).open('w') as f:done=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 assert done.returncode==0,(name,done.returncode)
 m=json.loads((a.output/name/'manifest.json').read_text());assert m['status']=='finished',m
records=[]
for path in sorted((a.output/'python').glob('profile-*.npz')):
 other=a.output/'compiled'/path.name
 with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
  assert float(x['t'])==float(y['t']) and int(x['step'])==int(y['step'])
  err=float(max(abs(x['q']-y['q']).ravel()));physical=float(max(abs(x['q_physical']-y['q_physical']).ravel()))
  assert err<1e-7 and physical<1e-7,(float(x['t']),err,physical)
  records.append(dict(time=float(x['t']),stored_error=err,physical_error=physical,python_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),compiled_sha256=hashlib.sha256(other.read_bytes()).hexdigest()))
review=dict(n=a.n,time=a.t,records=records,max_stored_difference=max(r['stored_error'] for r in records),max_physical_difference=max(r['physical_error'] for r in records),manifest_sha256={name:hashlib.sha256((a.output/name/'manifest.json').read_bytes()).hexdigest() for name in ['python','compiled']},scope='Implementation agreement through the failed wormhole evolution, not scientific admission or constraint repair')
(a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items() if k!='records'}))
