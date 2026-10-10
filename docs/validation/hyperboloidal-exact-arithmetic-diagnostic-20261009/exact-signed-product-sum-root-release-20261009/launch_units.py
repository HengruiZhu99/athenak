"""Preserve complete true lifecycle for fresh standalone Release/ASanUB units."""
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
release=load(HERE/'units-release.json');owner=Path(release['owner']);r=load(owner/'recipe.json')
out=HERE/'units-invocation001';out.mkdir(exist_ok=False);started=time.monotonic()
record={'completed':False,'passed':False,'returncode':None,'no_RWM_adoption_or_failure_upgrade':True}
pins=dict(release['pins']);pins[str(HERE/'units-authorization.json')]=release['authorization_sha256'];pins[str(HERE/'units-release.json')]=sha(HERE/'units-release.json')
try:
 assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
 verify(pins);write(out/'pins-before.json',pins)
 env=dict(os.environ)
 for key in r['sanitized_environment']:env.pop(key,None)
 env.update(r['environment'])
 command=[r['python']['path'],'-I','-B',str(owner/'run_gate.py'),'--authorization',str(HERE/'units-authorization.json')]
 write(out/'invocation.json',{'command':command,'environment':r['environment'],'sanitized':r['sanitized_environment'],'cwd':str(HERE.parents[1])})
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
  done=subprocess.run(command,cwd=HERE.parents[1],env=env,stdout=stdout,stderr=stderr,check=False,timeout=300)
 record['returncode']=done.returncode;child=owner/'attempts/units001/receipt.json'
 if child.exists():
  receipt=load(child);record['child_receipt_sha256']=sha(child)
  record['passed']=bool(done.returncode==0 and receipt.get('completed') is True and receipt.get('passed') is True and receipt.get('returncode')==0 and receipt.get('inputs_unchanged') is True)
  record['completed']=record['passed']
except BaseException as exc:record.update(error=repr(exc),traceback=traceback.format_exc())
finally:
 try:verify(pins);record['inputs_unchanged']=True
 except BaseException as exc:record.update(inputs_unchanged=False,passed=False,post_pin_failure=str(exc))
 record['seconds']=time.monotonic()-started
 record['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for folder in (out,owner/'attempts/units001') if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file()]
 write(out/'receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}))
if not record['passed']:raise SystemExit(1)
