#!/usr/bin/env python3
"""Single reviewed RF2 submit using Aurora's observed qstat -u text format.
The frozen v2 controller's JSON assumption failed before intent/qsub; this
entry point preserves its immutable plan, worker, science and terminal verifier.
"""
import sys,os,json,re,time,subprocess
from pathlib import Path
root=Path(sys.argv[1]).resolve();pf=root/sys.argv[2];sys.path.insert(0,str(root/'launcher-v3'))
from control import read,write,digest,bindings
from storage import capacity
plan=read(pf);bindings(root,plan);d=pf.parent;lock=root/'allocation-lock.json'
assert not lock.exists() and not (d/'submission.json').exists() and not (root/plan['output']).exists()
r=subprocess.run(['qstat','-u',os.environ['USER']],text=True,capture_output=True,check=True);(d/'reviewed-pre-submit-jobs.txt').write_text(r.stdout+r.stderr)
rows=[x for x in r.stdout.splitlines() if re.match(r'^\d+\.',x.strip())]
if any(re.search(r'\b[QR]\b',x) for x in rows):raise RuntimeError('A user job is queued/running; reconcile before this reviewed submit')
for q in ['debug','debug-scaling']:
 s=subprocess.check_output(['qstat','-Qf',q],text=True);(d/('reviewed-pre-submit-'+q+'.txt')).write_text(s)
 if q=='debug-scaling':assert 'enabled = True' in s and 'started = True' in s
cap=capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes'])
sub=dict(status='intent',time=time.time(),plan=str(pf.relative_to(root)),plan_sha256=digest(pf),storage=cap,submission_helper_sha256=digest(__file__),reason='Frozen v2 qstat -u ignores JSON request; observed text queue inspected without altering frozen scientific or worker bindings')
for p in [lock,d/'submission.json']:
 with p.open('x') as f:json.dump(sub,f)
command=['qsub','-o',str(d/'pbs.log'),str(d/'job.pbs')];r=subprocess.run(command,cwd=root,text=True,capture_output=True);write(d/'qsub-response.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
if r.returncode or not re.fullmatch(r'\d+(?:\.[\w.-]+)?',r.stdout.strip()):raise RuntimeError('qsub outcome ambiguous/rejected; keep intent, never retry automatically')
sub.update(status='submitted',job_id=r.stdout.strip());write(d/'submission.json',sub);write(lock,sub);print(json.dumps(sub))
