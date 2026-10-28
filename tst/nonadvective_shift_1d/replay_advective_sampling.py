"""Bounded diagnostic replay of a verified successful old trajectory.

Uses its frozen scientific code unchanged and proves exact recovery of its
completed .2 checkpoint. The original worktree and evidence remain untouched.
"""
import argparse,datetime,hashlib,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((a.reference/'manifest.json').read_text());assert m['status']=='completed'
for rel,sha in m['sources'].items():assert digest(a.reference/'source'/rel)==sha
libname=Path(m['arguments']['library']).name;assert digest(a.reference/'source'/libname)==m['library_sha256']
ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True);children=[line for line in ps.splitlines() if ('frozen_continuation.py --parent' in line or 'nonadvective_shift_1d/run_segment.py --campaign' in line) and '/bin/zsh -c' not in line]
assert len(children)<2,'No evolution slot available'
a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(a.reference/'source',a.output/'source')
record=dict(status='running',pid=os.getpid(),command=sys.argv,reference=str(a.reference),reference_manifest_sha256=digest(a.reference/'manifest.json'),library_sha256=m['library_sha256'],scientific_sources=m['sources'],wall_seconds_limit=420,scope='Fresh diagnostic replay of a successful reference interval, not restart of any failed or main-thread trajectory')
(a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
sys.path.insert(0,str(a.output.resolve()/'source/tst/hyperbolic_shift_1d'))
from puncture import PunctureEvolution
kwargs={k:v for k,v in m['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
e=PunctureEvolution(m['arguments']['n'],a.output.resolve()/'source'/libname,**kwargs);q=e.initial();dt=m['dt'];steps=round(.2/dt);assert abs(steps*dt-.2)<1e-13
history=[];times=[];states=[];physical=[];begin=time.monotonic()
for step in range(steps+1):
 if step:q=e.step(q,dt)
 if time.monotonic()-begin>415:raise TimeoutError('Bounded diagnostic replay')
 if step%256==0 or step==steps:
  t=step*dt;row=e.report(q,t);history.append(row);times.append(t);states.append(q.copy());x=e.last
  physical.append(np.c_[x['chi'],x['g'],x['physical'][:,2:6],x['alpha'],x['beta'],q[:,8],q[:,9]])
  assert row['max_relative_null']<5e-14 and row['pole']<.01
# Locate the actual completed successful checkpoint at .2 and verify its files.
case=a.reference
while True:
 cm=json.loads((case/'manifest.json').read_text())
 if abs(cm['t']-.2)<1e-13:break
 assert cm.get('parent');case=Path(cm['parent']['path'])
for name,sha in cm['checkpoint_hashes'].items():assert digest(case/name)==sha
with np.load(case/'profiles.npz') as z:
 assert np.array_equal(q,z['last']),'Replay does not exactly recover completed reference checkpoint'
 assert np.array_equal(e.compensation,z['compensation']),'Compensated update replay mismatch'
np.savez_compressed(a.output/'profiles.npz',r=e.r,t=times,q=states,physical=physical,last=q,last_time=.2,compensation=e.compensation)
(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
record.update(status='completed',t=.2,dt=dt,last_accepted_step=steps,wall_seconds=time.monotonic()-begin,stage_stats=e.stats,exact_checkpoint_recovery=dict(case=str(case),state=True,compensation=True),checkpoint_hashes={name:digest(a.output/name) for name in ['profiles.npz','history.json']})
(a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ['scientific_sources','stage_stats']},indent=2))
