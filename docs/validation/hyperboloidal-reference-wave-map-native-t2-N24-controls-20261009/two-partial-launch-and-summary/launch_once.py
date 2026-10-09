"""Exact two root-released partial observer invocations; no native evolution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent;R=P.parents[2]
PYTHON='/Library/Developer/CommandLineTools/usr/bin/python3'
ROWS=[('half','reference-wave-map-partial-half-N24-held-20261009','wave-map-native-half-N24-partial-root-release-20261009','af6ead0e960e172d1d88f82a059399a00afd2eb309131c1b1157f9fa9e0ccbbe','c0483e7cf11af63d843dd0948c545be3bfa016d8a012e5b1109dc59291db169e'),('small','reference-wave-map-partial-small-N24-held-20261009','wave-map-native-small-N24-partial-root-release-20261009','1351fef890c4da698c9bc089c71b371d1a75ac59837f2cf83b8ca4c1afb5da6c','f81cc68e0b6e88b1977ae489f9dc6ca4993ae15246c2ceb93f08f216e743b12e')]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def one(row):
 label,prefix,authprefix,expectedauth,expectedrecipe=row;D=P/label;D.mkdir(exist_ok=False)
 Q=R/'build-layer-research'/prefix;A=R/'build-layer-research'/authprefix/'authorization.json';S=Q/'run_released_observation001.py'
 envs={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(R/'build-layer-research/boundary/python-deps')}
 command=[PYTHON,'-B',str(S),str(A),expectedauth]
 record={'partial_diagnostic_only':True,'accepted_native_run':False,'case_label':label,'command':command,'cwd':str(R),'environment_overrides':envs,'root_authorization_sha256':expectedauth,'completed_outer_invocation':False,'returncode':None,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()};started=time.monotonic()
 dump(D/'receipt.json',record)
 try:
  assert sha(A)==expectedauth and sha(Q/'recipe.json')==expectedrecipe
  assert sha(S)=='5aec4a6a6ab5acf3f633a9589cb92c01fb90514228d4f604920dfb0fff4a036f'
  assert sha(Q/'observe_partial.py')=='c7ada0324bcd91c1e359620f46efd6614286876810a9d692a98ef6e37cb3a0f3'
  protected={str(p):sha(p) for p in [A,S,Q/'recipe.json',Q/'observe_partial.py',Q/'source-index.json',Path(__file__).resolve(),Path(PYTHON)]}
  dump(D/'pins-before.json',protected);env=os.environ.copy();env.update(envs)
  so=D/'stdout';se=D/'stderr'
  with so.open('wb') as f,se.open('wb') as g:r=subprocess.run(command,cwd=R,env=env,stdout=f,stderr=g)
  record.update(returncode=r.returncode,stdout=str(so),stdout_sha256=sha(so),stderr=str(se),stderr_sha256=sha(se),completed_outer_invocation=True)
  for p,s in protected.items():assert sha(p)==s,p
  dump(D/'pins-after.json',protected);record['protected_unchanged']=True
 except Exception as exc:record['outer_failure']=repr(exc)
 record['seconds']=time.monotonic()-started;dump(D/'receipt.json',record);return record
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,ROWS))
dump(P/'receipt.json',{'scope':'Exactly two root-released partial diagnostics; no native advance.','source_sha256':sha(__file__),'results':results,'all_outer_commands_returned_zero':all(q.get('returncode')==0 for q in results)})
print(json.dumps(results,indent=2))
assert all(q.get('returncode')==0 for q in results),'Preserved partial diagnostic protocol failure; no retries.'
