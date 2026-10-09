"""Held invocation wrapper. Requires exact root authorization path and SHA on CLI."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
P=Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
A=Path(sys.argv[1]).resolve();expected_auth=sys.argv[2]
OUT=P/'invocations001';OUT.mkdir(exist_ok=False)
record={'partial_diagnostic_only':True,'accepted_native_run':False,'observer_completed':False,'outer_wrapper_completed':False,'scope':'Exactly one original failed N24 process; no native advance or completed analyzer.'}
started=time.monotonic();protected={str(Path(__file__).resolve()):sha(__file__),str(A):sha(A)}
try:
 assert sha(A)==expected_auth,'root authorization SHA differs'
 idx=load(P/'source-index.json')
 for item in idx['files']:
  assert sha(item['path'])==item['sha256'],item['path'];protected[item['path']]=item['sha256']
 Q=P/'recipe.json';S=P/'observe_partial.py';q=load(Q);a=load(A)
 assert a['partial_native_snapshot_observation_authorized'] is True
 assert a['observer_sha256']==sha(S) and a['recipe_sha256']==sha(Q)
 assert set(a['cases'])==set(q['cases']) and len(q['cases'])==1
 name=next(iter(q['cases']));protected.update(q['fixed_pins'])
 for p,s in protected.items():assert sha(p)==s,p
 dump(OUT/'pins-before.json',protected)
 python='/Library/Developer/CommandLineTools/usr/bin/python3'
 overrides={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(R/'build-layer-research/boundary/python-deps')}
 env=os.environ.copy();env.update(overrides)
 command=[python,'-B',str(S),str(A),name]
 context={'command':command,'cwd':str(R),'environment_overrides':overrides,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'case':name,'root_release_sha256':sha(A),'scope':record['scope']}
 dump(OUT/'invocation-context.json',context);(OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
 so=OUT/'observer.stdout';se=OUT/'observer.stderr'
 with so.open('wb') as f,se.open('wb') as g:result=subprocess.run(command,cwd=R,env=env,stdout=f,stderr=g)
 record.update(context,returncode=result.returncode,stdout=str(so),stdout_sha256=sha(so),stderr=str(se),stderr_sha256=sha(se))
 for p,s in protected.items():assert sha(p)==s,p
 dump(OUT/'pins-after.json',protected)
 record.update(pins_unchanged=True,outer_wrapper_completed=True,observer_completed=result.returncode==0)
except Exception as exc:record['protocol_error']=repr(exc)
record['seconds']=time.monotonic()-started;dump(OUT/'receipt.json',record);print(json.dumps(record),flush=True)
assert record['outer_wrapper_completed'] and record.get('returncode')==0,'partial diagnostic wrapper stopped; never native acceptance'
