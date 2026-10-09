"""Preserve true child lifecycle and all errors for separately released stages."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(pins):
 for p,h in pins.items():assert sha(p)==h,p
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['controls','qualification']);args=ap.parse_args()
release=load(HERE/'release.json');SRC=Path(release['owner']);recipe=load(SRC/'control-recipe.json')
stage=args.stage;auth=HERE/(stage+'-authorization.json');out=HERE/(stage+'-invocation001');out.mkdir(exist_ok=False)
started=time.monotonic();record={'completed':False,'accepted_stage':False,'returncode':1,'stage':stage,'original_full_gate_remains_failed':True}
pins=dict(release['pins']);pins[str(auth)]=release['authorizations'][stage]
for p in ('release.json','source-review001.json','prepare_release001.py','launch.py'):pins[str(HERE/p)]=sha(HERE/p)
try:
 assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
 verify(pins);write(out/'pins-before.json',pins)
 env=dict(os.environ)
 for k in ('PYTHONPATH','PYTHONHOME','PYTHONWARNINGS'):env.pop(k,None)
 env.update(recipe['required_environment']);env['OMP_NUM_THREADS']='1'
 script='controls_only.py' if stage=='controls' else 'qualify_saved_checks.py'
 dest=Path(recipe['fresh_control_output' if stage=='controls' else 'fresh_saved_qualification_output'])
 command=[recipe['held_interpreter_command'],'-B','-s',str(SRC/script),'--recipe',str(SRC/'control-recipe.json'),'--authorization',str(auth),'--output',str(dest)]
 write(out/'invocation.json',{'command':command,'cwd':str(REPO),'environment':{k:env[k] for k in list(recipe['required_environment'])+['OMP_NUM_THREADS']},'unset':['PYTHONPATH','PYTHONHOME','PYTHONWARNINGS']})
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
  child=subprocess.run(command,cwd=REPO,env=env,stdout=stdout,stderr=stderr,check=False)
 record['returncode']=child.returncode
 if (dest/'receipt.json').exists():
  receipt=load(dest/'receipt.json');field='passed_corrected_control_gate' if stage=='controls' else 'passed_saved_noncontrol_check_qualification'
  record['child_receipt_sha256']=sha(dest/'receipt.json')
  record['accepted_stage']=bool(child.returncode==0 and receipt.get(field) is True and receipt.get('sources_unchanged') is True)
  if stage=='controls':record['counts']={k:receipt.get(k) for k in ('checks','control_rows','completed_control_roots')}
  else:record['counts']={k:receipt.get(k) for k in ('retained_saved_checks','retained_saved_failures','entire_control_slice_deferred')}
 record['completed']=record['accepted_stage']
except BaseException as exc:
 record['failure']=type(exc).__name__+': '+str(exc);(out/'failure.txt').write_text(traceback.format_exc())
finally:
 try:verify(pins);record['inputs_unchanged']=True
 except BaseException as exc:record.update(inputs_unchanged=False,accepted_stage=False,post_pin_failure=str(exc))
 record['seconds']=time.monotonic()-started
 record['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for folder in (out,SRC/('control-attempt001' if stage=='controls' else 'saved-qualification001')) if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file()]
 write(out/'receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}))
if not record['accepted_stage']:raise SystemExit(1)
