#!/usr/bin/env python3
"""Copy a closed early snapshot without treating an active run as complete."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--time',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing prefix evidence is immutable')
name=f'fields-t{a.time:.3f}.f64';host='hzhu@aurora.alcf.anl.gov';options=['-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C'];ssh=['ssh',*options,host];program='run='+repr(a.run)+'\ntime='+repr(a.time)+'\nname='+repr(name)+'\n'+'''import hashlib,json
from pathlib import Path
r=Path(run)
assert any(line.startswith('t='+format(time,'g')+' ') for line in (r/'run.log').read_text().splitlines()),'Native snapshot completion report absent'
files={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in [name,'parameters.json','invocation.json']}
stat=(r/name).stat()
print(json.dumps(dict(run=run,time=time,files=files,size=stat.st_size,mtime_ns=stat.st_mtime_ns,complete=False,scope='Closed native field snapshot only; active evolution and end-of-run frozen integrity remain outstanding.')))
'''
r=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True,check=True);record=json.loads(r.stdout);a.output.mkdir(parents=True);(a.output/'remote-prefix-before.json').write_text(json.dumps(record,indent=2)+'\n')
for file in record['files']:subprocess.run(['scp',*options,host+':'+shlex.quote(a.run+'/'+file),str(a.output/file)],check=True)
after=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True,check=True);last=json.loads(after.stdout);passed=last==record and all(sha(a.output/file)==digest for file,digest in record['files'].items());(a.output/'prefix-collection-receipt.json').write_text(json.dumps(dict(passed=passed,admission=False,native_run_complete=False,remote=record,local_files={file:sha(a.output/file) for file in record['files']},source_sha256=sha(Path(__file__)),collected=time.time()),indent=2)+'\n');print(json.dumps(dict(passed=passed,admission=False,time=a.time),indent=2))
if not passed:raise RuntimeError('Closed snapshot changed during collection')
