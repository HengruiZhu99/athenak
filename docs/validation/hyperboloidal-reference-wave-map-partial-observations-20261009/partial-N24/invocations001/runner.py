from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent
R=P.parents[1]
A=R/'build-layer-research/wave-map-native-N24-partial-root-release-20261009/authorization.json'
S=P/'observe_partial.py';Q=P/'recipe.json';OUT=P/'invocations001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
expected={str(A):'d2d9d09023c82063790c8a10db5cb3d174364df028a80f68ebe9430f02f06747',str(S):'c7ada0324bcd91c1e359620f46efd6614286876810a9d692a98ef6e37cb3a0f3',str(Q):'9228f05a83abbb3f0e6c047946ced7d8ffd37509cae64f058d431780627b80d6'}
assert all(sha(p)==s for p,s in expected.items())
OUT.mkdir(exist_ok=False)
python='/Library/Developer/CommandLineTools/usr/bin/python3'
overrides={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(R/'build-layer-research/boundary/python-deps')}
env=os.environ.copy();env.update(overrides)
protected=dict(expected);protected[str(Path(__file__))]=sha(__file__);protected[python]=sha(python)
dump(OUT/'pins-before.json',protected)
dump(OUT/'invocation-context.json',{'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'cwd':str(R),'python':python,'environment_overrides':overrides,'scope':'Released failed-N24 observations only; no native advance or acceptance.'})
(OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
command=[python,'-B',str(S),str(A),'wave-map-N24-large-t2']
so=OUT/'observer.stdout';se=OUT/'observer.stderr';started=time.monotonic()
with so.open('wb') as f,se.open('wb') as g:result=subprocess.run(command,cwd=R,env=env,stdout=f,stderr=g)
record={'command':command,'cwd':str(R),'environment_overrides':overrides,'returncode':result.returncode,'seconds':time.monotonic()-started,'stdout':str(so),'stdout_sha256':sha(so),'stderr':str(se),'stderr_sha256':sha(se),'accepted_native_run':False,'partial_diagnostic_only':True}
assert all(sha(p)==s for p,s in protected.items())
dump(OUT/'pins-after.json',protected);record['pins_unchanged']=True
dump(OUT/'receipt.json',record);print(json.dumps(record),flush=True)
assert result.returncode==0
