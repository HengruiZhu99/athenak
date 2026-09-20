"""One explicit submission with persistent uncertainty protection."""
from pathlib import Path
import datetime,fcntl,json,subprocess
from check_package import check
p=Path(__file__).resolve().parent
with (p/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=p/'submission.json'
 if state.exists():raise RuntimeError('Existing submission record; reconcile instead of retrying')
 checked=check(True)
 q=subprocess.run(['qstat','-f','-F','json','-u','hzhu'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,check=True)
 (p/'user-jobs-before-submit.json').write_text(q.stdout)
 jobs=json.loads(q.stdout).get('Jobs',{})
 if any(j.get('Job_Name')=='strongfield_smr232' and j.get('job_state')not in['F','X']for j in jobs.values()):raise RuntimeError('An active matching diagnostic already exists')
 record={'phase':'submitting','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_sha256':checked['package_sha256'],'pbs':'submit.pbs','account':'MHDTidal','queue':'debug','nodes':2,'ranks':24}
 def save():state.write_text(json.dumps(record,indent=2)+'\n')
 save()
 try:
  r=subprocess.run(['qsub','submit.pbs'],cwd=p,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
  (p/'qsub.stdout').write_text(r.stdout);(p/'qsub.stderr').write_text(r.stderr)
  record.update(qsub_exit=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
  job=r.stdout.strip()
  if r.returncode!=0 or not job.split('.')[0].isdigit():
   record['phase']='submission_uncertain';save();raise RuntimeError('Submission outcome requires reconciliation')
  record.update(phase='submitted',job_id=job);save();print(json.dumps(record,indent=2))
 except Exception:
  if record['phase']=='submitting':record['phase']='submission_uncertain';save()
  raise
