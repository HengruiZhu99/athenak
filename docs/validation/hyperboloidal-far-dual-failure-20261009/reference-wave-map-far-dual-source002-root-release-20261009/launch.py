from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,traceback
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
  if sha(p)!=d:raise RuntimeError('changed pin '+p)
ap=argparse.ArgumentParser();ap.add_argument('build',choices=['release','debug']);args=ap.parse_args()
release=load(HERE/'release.json');owner=Path(release['owner']);recipe=load(owner/'recipe.json')
if args.build=='debug':
 previous=load(HERE/'release-invocation001/receipt.json')
 if previous.get('accepted_local_gate') is not True:raise RuntimeError('actual successful Release prerequisite required')
out=HERE/(args.build+'-invocation001');out.mkdir(exist_ok=False);started=time.monotonic();record=dict(completed=False,returncode=None,accepted_local_gate=False,build=args.build)
pins=dict(release['pins']);pins[str(HERE/'authorization.json')]=release['authorization_sha256'];pins[str(HERE/'launch.py')]=sha(__file__)
try:
 verify(pins);write(out/'pins-before.json',pins)
 env=os.environ.copy();env.update(recipe['environment']);env['PYTHONOPTIMIZE']='0';env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None);env.pop('PYTHONWARNINGS',None)
 cmd=[recipe['python'],'-I','-B',str(owner/'run_once.py'),'--authorization',str(HERE/'authorization.json'),'--build',args.build]
 write(out/'invocation.json',dict(command=cmd,environment={k:env[k] for k in list(recipe['environment'])+['PYTHONOPTIMIZE']},cwd=recipe['repository']))
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:done=subprocess.run(cmd,cwd=recipe['repository'],env=env,stdout=stdout,stderr=stderr,check=False)
 record['returncode']=done.returncode
 child=owner/'attempts'/recipe['attempt_names'][args.build]/'receipt.json'
 if child.exists():
  receipt=load(child);record['child_receipt_sha256']=sha(child)
  record.update(completed=receipt.get('completed') is True,accepted_local_gate=bool(done.returncode==0 and receipt.get('passed') is True and receipt.get('returncode')==0 and receipt.get('source_inputs_unchanged') is True))
except BaseException as e:
 record['exception']=repr(e);(out/'failure.txt').write_text(traceback.format_exc())
finally:
 try:verify(pins);record['inputs_unchanged']=True
 except BaseException as e:record.update(inputs_unchanged=False,post_pin_exception=repr(e),accepted_local_gate=False)
 record['seconds']=time.monotonic()-started
 record['output_inventory']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for folder in [out,owner/'attempts'/recipe['attempt_names'][args.build]] if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file()]
 write(out/'receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}));raise SystemExit(0 if record['accepted_local_gate'] else 1)
