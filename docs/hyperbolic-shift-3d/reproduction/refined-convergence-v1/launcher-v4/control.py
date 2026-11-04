#!/usr/bin/env python3
"""Immutable RF2 plan, single submission, and terminal admission; no numerical retries."""
import argparse, hashlib, json, math, os, pathlib, re, subprocess, time
from storage import capacity
P=pathlib.Path

def digest(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def write(p,v):
 p=P(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def read(p):return json.loads(P(p).read_text())
def bindings(root,plan):
 for f,h in plan['bindings'].items():
  p=(root/f).resolve()
  if not p.is_relative_to(root) or digest(p)!=h:raise RuntimeError('binding mismatch '+f)
def verify(root,plan,out):
 bindings(root,plan);report=read(out/'state.report.json')
 if report['status'] not in ('completed','checkpointed') or not report.get('checkpoint_admitted'):raise RuntimeError('unadmitted native state')
 cp=out/'state.checkpoint'
 if not os.access(root/'launcher-v4/verify-checkpoint',os.X_OK):raise RuntimeError('verifier is not executable')
 v=json.loads(subprocess.check_output([str(root/'launcher-v4/verify-checkpoint'),str(cp)],text=True))
 for k in ('identity','n','steps','time','dt','leaves'):
  if v[k]!=report[k]:raise RuntimeError('checkpoint/report disagreement '+k)
 if v['identity']!=plan['identity'] or v['n']!=plan['n'] or v['kind']!=2:raise RuntimeError('scientific configuration mismatch')
 if report['ranks']!=plan['ranks'] or report['backend']!='SYCL':raise RuntimeError('backend/rank mismatch')
 if report['requested_dt']!=plan['dt'] or report['target_time']!=plan['target_time'] or report['diagnostic_interval']!=plan['diagnostic_interval']:raise RuntimeError('requested integration policy mismatch')
 if not (plan['start_time']<v['time']<=plan['target_time']+1e-12 and v['steps']>plan['start_steps']):raise RuntimeError('no forward progress or overshoot')
 if abs(v['time']-min(v['steps']*plan['dt'],plan['target_time']))>2e-10:raise RuntimeError('step/time mismatch')
 if report['status']=='completed':
  if abs(v['time']-plan['target_time'])>1e-12:raise RuntimeError('premature completion')
 else:
  if v['time']>=plan['target_time'] or abs(v['dt']-plan['dt'])>1e-14:raise RuntimeError('invalid walltime endpoint')
 if report['checkpoint_bytes']!=cp.stat().st_size or v['bytes']!=cp.stat().st_size:raise RuntimeError('length mismatch')
 rows=[json.loads(x) for x in (out/'state.history.jsonl').read_text().splitlines() if x.strip()]
 if not rows or abs(rows[-1]['time']-v['time'])>1e-12 or rows[-1]['step']!=v['steps'] or rows[-1]['tracker']!=v['tracker']:raise RuntimeError('history endpoint mismatch')
 if any(abs(row['penalty']-.6510416666660156)>1e-12 for row in rows):raise RuntimeError('effective scri penalty mismatch')
 if any(not math.isfinite(row[k]) for row in rows for k in ('H','M','Mr','null','ghost_null','cfl')):raise RuntimeError('nonfinite history')
 v.update(sha256=digest(cp),status=report['status'],source_commit=plan['source_commit'],plan_sha256=digest(root/plan['plan']),native_report_sha256=digest(out/'state.report.json'))
 return v

def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['submit','collect']);a.add_argument('root');a.add_argument('plan');args=a.parse_args()
 root=P(args.root).resolve();pf=(root/args.plan).resolve();plan=read(pf);directory=pf.parent;out=root/plan['output'];bindings(root,plan)
 lock=root/'allocation-lock.json'
 if args.action=='submit':
  if out.exists() or (directory/'submission.json').exists() or lock.exists():raise RuntimeError('existing output/submission/lock; reconcile rather than duplicate')
  jobs=subprocess.run(['qstat','-f','-F','json','-u',os.environ['USER']],capture_output=True,text=True,check=True)
  (directory/'pre-submit-jobs.txt').write_text(jobs.stdout+jobs.stderr)
  # Preserve other allocations; a single queued RF2 run may wait behind a running BN1 job.
  if any(j.get('queue') in ('debug','debug-scaling') and j.get('job_state')=='Q' for j in json.loads(jobs.stdout).get('Jobs',{}).values()):raise RuntimeError('user already has a queued debug job')
  for q in ('debug','debug-scaling'):
   s=subprocess.check_output(['qstat','-Qf',q],text=True);(directory/('pre-submit-'+q+'.txt')).write_text(s)
   if q=='debug-scaling' and not ('enabled = True' in s and 'started = True' in s):raise RuntimeError('queue disabled')
  cap=capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes'])
  intent=dict(status='intent',time=time.time(),plan=str(pf.relative_to(root)),plan_sha256=digest(pf),storage=cap)
  with lock.open('x') as f:json.dump(intent,f)
  with (directory/'submission.json').open('x') as f:json.dump(intent,f)
  cmd=['qsub','-o',str(directory/'pbs.log'),str(directory/'job.pbs')]
  r=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
  write(directory/'qsub-response.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
  if r.returncode or not re.fullmatch(r'\d+(?:\.[\w.-]+)?',r.stdout.strip()):raise RuntimeError('unknown/rejected qsub outcome; keep intent and reconcile')
  intent.update(status='submitted',job_id=r.stdout.strip());write(directory/'submission.json',intent);write(lock,intent);print(json.dumps(intent))
 else:
  sub=read(directory/'submission.json');q=subprocess.run(['qstat','-xf',sub['job_id']],text=True,capture_output=True)
  (directory/'terminal-qstat.txt').write_text(q.stdout+q.stderr)
  if q.returncode or not re.search(r'job_state\s*=\s*F\b',q.stdout):raise RuntimeError('not terminal; leave lock')
  exit_match=re.search(r'Exit_status\s*=\s*(-?\d+)',q.stdout)
  if not exit_match:raise RuntimeError('missing scheduler exit')
  state=read(out/'worker.json');native=state.get('native_returncode');pbs=int(exit_match[1])
  for f,b in state['artifacts'].items():
   p=out/f
   if digest(p)!=b['sha256'] or p.stat().st_size!=b['bytes']:raise RuntimeError('artifact changed '+f)
  result=dict(job_id=sub['job_id'],scheduler_exit=pbs,worker=state['status'],native_returncode=native,terminal=True,admitted=False)
  if state['status'] in ('completed','checkpointed'):
   expected=0 if state['status']=='completed' else 3
   if pbs!=expected or native!=expected:raise RuntimeError('scheduler/worker/native mismatch')
   result['checkpoint']=verify(root,plan,out);result['admitted']=True
  elif pbs==0:raise RuntimeError('unexpected successful scheduler with failed worker')
  write(directory/'terminal-verification.json',result)
  if lock.exists():
   l=read(lock)
   if l.get('job_id')!=sub['job_id'] or l['plan_sha256']!=digest(pf):raise RuntimeError('different allocation lock')
   write(directory/'allocation-released.json',dict(job_id=sub['job_id'],plan_sha256=digest(pf),terminal_verification_sha256=digest(directory/'terminal-verification.json'),time=time.time()));lock.unlink()
  print(json.dumps(result))
if __name__=='__main__':main()
