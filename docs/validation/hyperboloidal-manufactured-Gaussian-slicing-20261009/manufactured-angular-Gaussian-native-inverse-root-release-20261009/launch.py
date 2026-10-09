from pathlib import Path
import hashlib,json,subprocess,os,time,traceback
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
r=load(HERE/'release.json');owner=Path(r['owner']);recipe=load(owner/'recipe.json');pins=dict(r['pins']);pins[str(HERE/'authorization.json')]=r['authorization_sha256'];pins[str(HERE/'launch.py')]=sha(__file__)
out=HERE/'invocation001';out.mkdir(exist_ok=False);record=dict(completed=False,accepted_consistency_execution=False);started=time.monotonic()
def verify():
 for p,d in pins.items():
  if sha(p)!=d:raise RuntimeError('changed '+p)
try:
 verify();write(out/'pins-before.json',pins)
 env=os.environ.copy();env.update(recipe['environment'])
 for k in ['PYTHONPATH','PYTHONHOME','PYTHONWARNINGS']:env.pop(k,None)
 cmd=[recipe['python_runtime_path'],'-I','-B',str(owner/'launch_once.py'),'--authorization',str(HERE/'authorization.json'),'--authorization-sha256',r['authorization_sha256']]
 write(out/'invocation.json',dict(command=cmd,environment=recipe['environment'],cwd=str(owner)))
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:done=subprocess.run(cmd,cwd=owner,env=env,stdout=stdout,stderr=stderr,check=False)
 record['returncode']=done.returncode
 receipt=load(owner/'outer-invocation001/receipt.json');record['owner_receipt_sha256']=sha(owner/'outer-invocation001/receipt.json')
 record.update(completed=receipt.get('completed') is True,accepted_consistency_execution=bool(done.returncode==0 and receipt.get('accepted_consistency_execution') is True and receipt.get('inputs_unchanged') is True),all_sampled_D_positive=receipt.get('all_sampled_D_positive'))
except BaseException as e:
 record['exception']=repr(e);(out/'failure.txt').write_text(traceback.format_exc())
finally:
 try:verify();record['inputs_unchanged']=True
 except BaseException as e:record.update(inputs_unchanged=False,accepted_consistency_execution=False,post_pin_failure=repr(e))
 record['seconds']=time.monotonic()-started
 record['output_inventory']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for folder in [out,owner/'outer-invocation001',owner/'attempt001'] if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file()]
 write(out/'receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}));raise SystemExit(0 if record['accepted_consistency_execution'] else 1)
