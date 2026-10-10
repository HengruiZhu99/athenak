"""One fresh local observational diagnostic with complete enclosing receipt."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify(pins):
 for p,h in pins.items():assert sha(p)==h,p
release=load(HERE/'release.json');owner=Path(release['owner']);recipe=load(owner/'recipe.json')
out=HERE/'diagnostic-invocation001';out.mkdir(exist_ok=False);started=time.monotonic()
record={'completed':False,'accepted_observational_diagnostic':False,'returncode':None,'original_far_Release_passed':False}
pins=dict(release['pins']);pins[str(HERE/'authorization.json')]=release['authorization_sha256'];pins[str(HERE/'release.json')]=sha(HERE/'release.json')
try:
 assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
 verify(pins);write(out/'pins-before.json',pins)
 env=dict(os.environ)
 for key in ('PYTHONHOME','PYTHONPATH','PYTHONWARNINGS'):env.pop(key,None)
 env.update(recipe['environment']);env.update(PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1')
 command=[recipe['python'],'-I','-B',str(owner/'run_once.py'),'--authorization',str(HERE/'authorization.json'),'--authorization-sha256',sha(HERE/'authorization.json')]
 write(out/'invocation.json',{'command':command,'cwd':recipe['repository'],'environment':{**recipe['environment'],'PYTHONOPTIMIZE':'0','OMP_NUM_THREADS':'1'}})
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
  done=subprocess.run(command,cwd=recipe['repository'],env=env,stdout=stdout,stderr=stderr,check=False,timeout=300)
 record['returncode']=done.returncode;child=Path(recipe['attempt'])/'receipt.json'
 if child.exists():
  receipt=load(child);record['child_receipt_sha256']=sha(child)
  record['accepted_observational_diagnostic']=bool(done.returncode==0 and receipt.get('completed') is True and receipt.get('diagnostic_completed') is True and receipt.get('inputs_unchanged') is True and receipt.get('returncode')==0 and receipt.get('original_far_Release_passed') is False)
  record['completed']=record['accepted_observational_diagnostic']
except BaseException as exc:record.update(error=repr(exc),traceback=traceback.format_exc())
finally:
 try:verify(pins);record['inputs_unchanged']=True
 except BaseException as exc:record.update(inputs_unchanged=False,accepted_observational_diagnostic=False,post_pin_failure=str(exc))
 record['seconds']=time.monotonic()-started
 record['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for folder in (out,Path(recipe['attempt'])) if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file()]
 write(out/'receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}))
if not record['accepted_observational_diagnostic']:raise SystemExit(1)
