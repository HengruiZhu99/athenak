#!/usr/bin/env python3
"""One-shot validated migration. Never resubmit a failed validation/evolution."""
from pathlib import Path
import fcntl,json,subprocess,sys,time
root=Path(__file__).resolve().parent
campaign=Path('/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
job='58140008';source_sha='400f784f95e055ca3457953a463b00f03aa71a8b'
def status(phase,**extra):
 p=root/'recovery-lapse/pipeline-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,time=time.time(),validation_job=job,**extra),indent=2)+'\n');tmp.replace(p)
def run(script,*args):subprocess.run([sys.executable,str(root/script),*args],check=True)
with (root/'recovery-lapse/pipeline.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  if (campaign/'state.json').exists():raise RuntimeError('Campaign already has state; refuse duplicate')
  status('WAITING_FOR_VALIDATION')
  while True:
   live=subprocess.check_output(['squeue','-h','-u','hzhu','-o','%i'],text=True).split()
   if job not in live:break
   time.sleep(30)
  result=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
  rows=[r.split('|') for r in result.splitlines() if r.split('|')[0]==job]
  if len(rows)!=1 or rows[0][1:3]!=['COMPLETED','0:0']:raise RuntimeError('Validation allocation not successful: '+result)
  status('VERIFYING')
  run('verify_recovery_lapse.py')
  run('prepare_recovery_endpoint.py')
  run('prepare_recovery_campaign.py','--source-sha',source_sha)
  status('CAMPAIGN_STARTING')
  with (campaign/'controller.log').open('w') as log:
   subprocess.run([sys.executable,str(campaign/'controller.py'),'--baseline',str(root/'recovery-lapse/baseline'),'--sub','-.0485','--super','-.05','--adopt-sub',str(root/'recovery-lapse/endpoints/sub'),'--adopt-super',str(root/'early-lapse/historical-endpoints/super')],stdout=log,stderr=subprocess.STDOUT,check=True)
  status('CAMPAIGN_FINISHED')
 except BaseException as exc:
  status('FAILED',error=repr(exc));raise
