"""Unique-path execution receipt for the already released N8 matrix gate."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,datetime,sys
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--J',type=int,required=True);p.add_argument('--Q',type=int,required=True);p.add_argument('--name',required=True);p.add_argument('--theta',type=int,default=12);p.add_argument('--phi',type=int,default=24);a=p.parse_args()
d=P/a.name;d.mkdir(exist_ok=True);assert not (d/'run-receipt.json').exists() and not (d/'operator.npz').exists()
cmd=[sys.executable,str(P/'assemble_segmented.py'),'--J',str(a.J),'--N','8','--rb','.98','--Q',str(a.Q),'--theta',str(a.theta),'--phi',str(a.phi),'--name',a.name]
rec={'command':cmd,'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','PYTHONPATH')},'launch_HEAD':subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip(),'wrapper_sha256':sha(__file__),'source_sha256':sha(P/'assemble_segmented.py'),'executable_sha256':sha(P/'radial-bridge-release')}
(d/'run_assembly-launch-source.py').write_bytes(Path(__file__).read_bytes());(d/'launch-receipt.json').write_text(json.dumps(rec,indent=2)+'\n');start=time.monotonic()
with (d/'run.stdout').open('wb') as fo,(d/'run.stderr').open('wb') as fe:r=subprocess.run(cmd,stdout=fo,stderr=fe)
rec.update({'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-start,'stdout_sha256':sha(d/'run.stdout'),'stderr_sha256':sha(d/'run.stderr'),'source_unchanged':sha(P/'assemble_segmented.py')==rec['source_sha256'],'executable_unchanged':sha(P/'radial-bridge-release')==rec['executable_sha256']})
(d/'run-receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2));sys.exit(r.returncode)
