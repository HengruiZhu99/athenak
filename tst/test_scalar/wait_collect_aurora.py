#!/usr/bin/env python3
"""Read one scheduler record and collect exactly once after it finishes."""
import argparse,json,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--journal',type=Path,required=True);p.add_argument('--resume-watcher',action='store_true');a=p.parse_args()
if a.output.exists() or a.journal.exists()!=a.resume_watcher:raise RuntimeError('Existing collection is immutable; restarting an inspected scheduler journal requires --resume-watcher')
previous=None
if a.resume_watcher:
 for line in a.journal.read_text().splitlines():
  old=json.loads(line)
  if set(old['Jobs'])!={a.job}:raise RuntimeError('Scheduler journal belongs to another allocation')
  previous=old['Jobs'][a.job]['job_state']
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C','hzhu@aurora.alcf.anl.gov'];access_error=None
while True:
 r=subprocess.run([*ssh,'qstat','-xf','-F','json',a.job],capture_output=True,text=True)
 if r.returncode==255:
  message=r.stderr.strip()
  if message!=access_error:
   print('ACCESS UNAVAILABLE; existing allocation unchanged:',message,flush=True)
   with a.journal.with_suffix('.access.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),job=a.job,stderr=message))+'\n')
   access_error=message
  time.sleep(45);continue
 if r.returncode:raise RuntimeError('Cannot read scheduler state: '+r.stderr)
 if access_error is not None:print('ACCESS RESTORED; inspecting existing allocation',a.job,flush=True);access_error=None
 state=json.loads(r.stdout);record=state['Jobs'][a.job]
 with a.journal.open('a') as f:f.write(json.dumps(state)+'\n')
 if record['job_state']!=previous:print('STATE',record['job_state'],flush=True);previous=record['job_state']
 if previous=='F':break
 time.sleep(45)
subprocess.run([sys.executable,str(Path(__file__).with_name('collect_aurora.py')),'--root',a.root,'--job',a.job,'--output',str(a.output)],check=True)
