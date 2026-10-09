from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as s:s.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify(pins):
 for p,d in pins.items():
  if sha(p)!=d:raise RuntimeError('pin changed '+p)
release=load(HERE/'release.json');owner=Path(release['owner']);recipe=load(owner/'recipe.json')
out=HERE/'invocation001';out.mkdir(exist_ok=False);started=time.monotonic();record=dict(completed=False,returncode=None,accepted_finite_screen_identity=False)
pins=dict(release['pins']);pins[str(HERE/'authorization.json')]=release['authorization_sha256'];pins[str(HERE/'launch.py')]=sha(__file__)
try:
 verify(pins);write(out/'pins-before.json',pins)
 env=os.environ.copy();env.update(recipe['environment']);env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None);env.pop('PYTHONWARNINGS',None)
 cmd=[recipe['python_runtime_path'],'-I','-B',str(owner/'screen.py'),'--authorization',str(HERE/'authorization.json'),'--authorization-sha256',release['authorization_sha256'],'--output',str(owner/'attempts/screen001')]
 write(out/'invocation.json',dict(command=cmd,environment={k:env[k] for k in recipe['environment']},cwd=str(owner),scope='finite physical-event screen only'))
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:done=subprocess.run(cmd,cwd=owner,env=env,stdout=stdout,stderr=stderr,check=False)
 record['returncode']=done.returncode;receipt=load(owner/'attempts/screen001/receipt.json');record['child_receipt_sha256']=sha(owner/'attempts/screen001/receipt.json')
 if done.returncode==0 and receipt.get('completed') is True and receipt.get('inputs_unchanged') is True:
  result=load(owner/'attempts/screen001/result.json')
  accepted=(result.get('checks_passed') is True and result['samples_per_precision']==28032 and len(result['profiles'])==8 and all(q['samples']==3504 for q in result['profiles']) and result['global_positivity_proven'] is False and result['continuum_or_native_stability_accepted'] is False)
  record.update(completed=True,accepted_finite_screen_identity=accepted,all_sampled_D_positive=result['all_sampled_D_positive'],result_sha256=sha(owner/'attempts/screen001/result.json'))
except BaseException as e:
 record['exception']=repr(e);(out/'failure.txt').write_text(traceback.format_exc())
finally:
 try:verify(pins);record['inputs_unchanged']=True
 except BaseException as e:record.update(inputs_unchanged=False,post_pin_exception=repr(e),accepted_finite_screen_identity=False)
 record['seconds']=time.monotonic()-started
 record['output_inventory']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for folder in [out,owner/'attempts/screen001'] if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file()]
 write(out/'receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}));raise SystemExit(0 if record['accepted_finite_screen_identity'] else 1)
