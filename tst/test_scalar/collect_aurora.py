#!/usr/bin/env python3
"""Collect one already-completed scalar allocation without submitting or changing it."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--host',default='hzhu@aurora.alcf.anl.gov');p.add_argument('--transport',choices=['rsync','scp'],default='rsync');a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing collection is immutable; inspect its receipt')
 a.output.mkdir(parents=True);ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C',a.host]
 # This remote program reads only the isolated scalar result and source manifest.
 program='ROOT='+repr(a.root)+'\nJOB='+repr(a.job)+'\n'+'''import hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT);job=JOB;result=root/'results'/job
q=subprocess.run(['qstat','-xf','-F','json',job],capture_output=True,text=True)
if q.returncode:raise RuntimeError('Cannot confirm completed scheduler state: '+q.stderr)
scheduler=json.loads(q.stdout);record=scheduler['Jobs'][job]
if record['job_state']!='F':raise RuntimeError('Allocation is not finished; collection refused')
sources={};mismatches=[]
for line in (root/'source.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);name=name.strip().lstrip('*');sources[name]=digest
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:mismatches.append(name)
files={str(p.relative_to(result)):hashlib.sha256(p.read_bytes()).hexdigest() for p in result.rglob('*') if p.is_file()}
print(json.dumps(dict(scheduler=scheduler,source_files=sources,source_mismatches=mismatches,files=files,admission_sha256=hashlib.sha256((root/'cpu-admission.json').read_bytes()).hexdigest())))
'''
 r=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True)
 (a.output/'collection-access.log').write_text(r.stderr)
 if r.returncode:raise RuntimeError('Aurora collection failed; access log preserved')
 manifest=json.loads(r.stdout);(a.output/'remote-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if manifest['source_mismatches']:raise RuntimeError('Remote source hash mismatch; do not audit this as the submitted source')
 target=a.output/'result';remote_path=a.root+'/results/'+a.job
 if a.transport=='rsync':
  target.mkdir();command=['rsync','-az','--whole-file','--compress-level=3','--stats','-e',shlex.join(ssh[:-1]),a.host+':'+shlex.quote(remote_path+'/'),str(target)+'/']
 else:command=['scp','-r','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C',a.host+':'+shlex.quote(remote_path),str(target)]
 with (a.output/'collection-transfer.log').open('w') as log:copy=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 if copy.returncode:raise RuntimeError('Transfer failed; partial collection preserved')
 mismatch=[name for name,digest in manifest['files'].items() if not (target/name).is_file() or hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest]
 receipt=dict(job=a.job,host=a.host,remote_root=a.root,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=len(manifest['files']),mismatches=mismatch,passed=not mismatch,transport=a.transport,transfer_command=command,collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (a.output/'collection-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if mismatch:raise RuntimeError('Collected files do not match the remote SHA256 manifest')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
