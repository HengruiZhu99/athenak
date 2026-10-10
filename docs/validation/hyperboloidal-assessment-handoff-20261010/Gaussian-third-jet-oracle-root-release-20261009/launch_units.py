"""Run one reviewed units stage with an enclosing 60-second group cap."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(pins):
 for p,h in pins.items():
  if sha(p)!=h:raise RuntimeError('changed input '+p)
release=load(HERE/'units-release.json');owner=Path(release['owner']);recipe=load(owner/'recipe.json')
auth=HERE/'units-authorization.json';out=HERE/'units-invocation001';out.mkdir(exist_ok=False)
pins=dict(release['pins']);pins[str(auth)]=release['authorization_sha256']
start=time.monotonic();record={'completed':False,'accepted_stage':False,'stage':'units','returncode':None,'root_process_group_cap_seconds':60,'science_imports_in_root':False};process=None
try:
 verify(pins);write(out/'pins-before.json',pins)
 env=os.environ.copy()
 for name in ('PYTHONHOME','PYTHONPATH','PYTHONWARNINGS','PYTHONSTARTUP','PYTHONUSERBASE'):env.pop(name,None)
 env.update(recipe['environment'])
 cmd=[recipe['python_runtime_path'],'-I','-B',str(owner/'outer_once.py'),'--authorization',str(auth),'--authorization-sha256',release['authorization_sha256'],'--invocation',release['outer_output']]
 write(out/'invocation.json',{'command':cmd,'cwd':str(owner),'environment':recipe['environment'],'root_process_group_timeout_seconds':60})
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
  process=subprocess.Popen(cmd,cwd=owner,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  try:record['returncode']=process.wait(timeout=60)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGKILL);record['returncode']=process.wait();raise TimeoutError('root60second units group cap; partial evidence preserved')
 child=Path(release['output']);c=load(child/'receipt.json');r=load(child/'result.json');o=load(Path(release['outer_output'])/'receipt.json')
 accepted=bool(record['returncode']==0 and c.get('completed') and c.get('passed') and c.get('sources_unchanged') and c.get('stage')=='units' and r.get('passed') and r.get('checks')==318 and o.get('completed') and o.get('accepted_stage') and o.get('inputs_unchanged'))
 record.update(completed=accepted,accepted_stage=accepted,child_receipt_sha256=sha(child/'receipt.json'),result_sha256=sha(child/'result.json'),outer_receipt_sha256=sha(Path(release['outer_output'])/'receipt.json'))
except BaseException as e:
 record['error']=repr(e);(out/'failure.txt').open('x').write(traceback.format_exc())
finally:
 try:verify(pins);record['inputs_unchanged']=True
 except BaseException as e:record.update(inputs_unchanged=False,accepted_stage=False,post_pin_error=repr(e))
 record['elapsed_seconds']=time.monotonic()-start
 record['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'metadata_only':p.suffix.lower() in {'.jsonl','.npy','.npz'} or p.stat().st_size>1048576} for folder in (out,Path(release['outer_output']),Path(release['output'])) if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file()]
 write(out/'receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}))
raise SystemExit(0 if record['accepted_stage'] else 1)
