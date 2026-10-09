"""Capture unique auxiliary-readback command/output provenance without altering it."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--receipt-name',required=True);p.add_argument('script');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
d=P/a.receipt_name;assert not d.exists();d.mkdir();script=P/a.script;assert script.is_file()
cmd=[sys.executable,str(script),*a.arguments]
rec={'command':cmd,'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'launch_HEAD':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
     'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','PYTHONPATH')},
     'python_version':sys.version,'script_sha256':sha(script),'wrapper_sha256':sha(__file__)}
(d/'source.py').write_bytes(script.read_bytes());(d/'wrapper.py').write_bytes(Path(__file__).read_bytes())
(d/'launch.json').write_text(json.dumps(rec,indent=2)+'\n');start=time.monotonic()
with (d/'stdout').open('wb') as fo,(d/'stderr').open('wb') as fe:run=subprocess.run(cmd,stdout=fo,stderr=fe)
rec.update({'exit_code':run.returncode,'seconds':time.monotonic()-start,'stdout_sha256':sha(d/'stdout'),'stderr_sha256':sha(d/'stderr'),'script_unchanged':sha(script)==rec['script_sha256']})
(d/'receipt.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');print(json.dumps(rec,indent=2));sys.exit(run.returncode)
