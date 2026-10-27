#!/usr/bin/env python3
"""Bounded serial diagnostic matrix; never labels survival as acceptance."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);script=Path(__file__).with_name('evolve.py')
base=[sys.executable,str(script),'--library',a.library,'--scri-sat','4','--regularity-sat','8','--report-dt','.1']
commands=[];records=[]
def run(name,*args):
 out=a.output/name;cmd=base+['--output',str(out),*map(str,args)]
 commands.append(cmd);(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 r=subprocess.run(cmd,capture_output=True,text=True);(a.output/(name+'.log')).write_text(r.stdout+r.stderr)
 m=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {'status':'launch_failed'}
 rows=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()] if (out/'history.jsonl').exists() else []
 rec={'name':name,'returncode':r.returncode,'status':m['status'],'time':m.get('time'),'last_report':rows[-1] if rows else None,
  'max_scri_pole':max((x['scri_pole'] for x in rows),default=None),'max_scri_lapse_pole':max((x['scri_lapse_pole'] for x in rows),default=None)}
 records.append(rec);(a.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ['name','status','time','max_scri_pole']}),flush=True)
 return out
for n in [32,64,128]:run(f'pulse-n{n}-cfl05','--n',n,'--t',1,'--amplitude',.001,'--cfl',.05)
for cfl in [.025,.0125]:run(f'pulse-n64-cfl{cfl}','--n',64,'--t',1,'--amplitude',.001,'--cfl',cfl)
# Independently tighten quadrature/differentiation order; derivative order also
# changes, so this is sensitivity, not isolation of elliptic error alone.
run('pulse-n64-width9','--n',64,'--t',1,'--amplitude',.001,'--cfl',.05,'--width',9)
# Restart checks at aligned fixed-CFL times; no new numerical source changes.
dt=.05/63.5;end=128*dt
full=run('restart-full','--n',64,'--t',end,'--amplitude',.001)
first=run('restart-first','--n',64,'--t',64*dt,'--amplitude',.001)
second=run('restart-second','--n',64,'--t',end,'--amplitude',.001,'--resume',first/'checkpoint.npz')
import numpy as np
with np.load(full/'checkpoint.npz') as x,np.load(second/'checkpoint.npz') as y:
 check={'state_bitwise':np.array_equal(x['q'],y['q']),'time_bitwise':np.array_equal(x['t'],y['t']),'steps_equal':int(x['step'])==int(y['step'])}
(a.output/'restart-review.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check),flush=True)
