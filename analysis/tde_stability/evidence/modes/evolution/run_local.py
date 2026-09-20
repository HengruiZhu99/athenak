from pathlib import Path
import os,subprocess,json,hashlib,datetime
b=Path(__file__).resolve().parent
exe=b.parents[1]/'stability-isolation-20260919/athena-before-source-order'
# The executable predates optional operators; no source-order/Hamiltonian experiment is enabled.
assert exe.exists(),exe
work=b/'cpu-small-dx025';work.mkdir(exist_ok=True)
assert not (work/'run.log').exists()
cmd=[str(exe),'-i',str(b/'small_dx025.athinput'),'-d',str(work),'-t','00:20:00']
meta={'command':cmd,'OMP_NUM_THREADS':2,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'status':'running'}
(work/'manifest.json').write_text(json.dumps(meta,indent=2))
env=os.environ.copy();env['OMP_NUM_THREADS']='2'
with (work/'run.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
meta.update(status='completed',exit_code=p.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(work/'manifest.json').write_text(json.dumps(meta,indent=2));(work/'exit_code.txt').write_text(str(p.returncode)+'\n');print(json.dumps(meta,indent=2))
