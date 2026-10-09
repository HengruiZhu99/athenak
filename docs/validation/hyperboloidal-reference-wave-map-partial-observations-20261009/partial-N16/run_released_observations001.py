from pathlib import Path
import hashlib, json, os, subprocess, sys, time
P=Path(__file__).resolve().parent
R=P.parents[1]
A=R/'build-layer-research/wave-map-native-partial-root-release-20261009/authorization.json'
S=P/'observe_partial.py'
Q=P/'recipe.json'
OUT=P/'invocations001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
expected={str(A):'fe870d21c17cea1362c36cbf6387b0a8a4a5e731338815b2fc2e5b5d4c8c436e',str(S):'c7ada0324bcd91c1e359620f46efd6614286876810a9d692a98ef6e37cb3a0f3',str(Q):'070be42ffaf40f6d8eeba5b35908b143b2a6f74289ceaf376f02ea232f946e93'}
assert all(sha(p)==s for p,s in expected.items())
OUT.mkdir(exist_ok=False)
env=os.environ.copy()
overrides={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(R/'build-layer-research/boundary/python-deps')}
env.update(overrides)
python='/Library/Developer/CommandLineTools/usr/bin/python3'
protected=dict(expected);protected[str(Path(__file__))]=sha(__file__);protected[python]=sha(python)
dump(OUT/'pins-before.json',protected)
dump(OUT/'invocation-context.json',{'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'cwd':str(R),'python':python,'environment_overrides':overrides,'scope':'Authorized observations only; no native advance or acceptance.'})
(Path(OUT/'runner.py')).write_bytes(Path(__file__).read_bytes())
records=[]
for case in ['wave-map-N16-large-t2','c0-N16-large-t2']:
    command=[python,'-B',str(S),str(A),case]
    started=time.monotonic()
    so=OUT/(case+'.stdout');se=OUT/(case+'.stderr')
    with so.open('wb') as f, se.open('wb') as g:
        result=subprocess.run(command,cwd=R,env=env,stdout=f,stderr=g)
    records.append({'case':case,'command':command,'cwd':str(R),'environment_overrides':overrides,'returncode':result.returncode,'seconds':time.monotonic()-started,'stdout':str(so),'stdout_sha256':sha(so),'stderr':str(se),'stderr_sha256':sha(se),'accepted_native_run':False,'partial_diagnostic_only':True})
    dump(OUT/(case+'-invocation.json'),records[-1])
    print(json.dumps(records[-1]),flush=True)
assert all(sha(p)==s for p,s in protected.items())
dump(OUT/'pins-after.json',protected)
dump(OUT/'receipt.json',{'calls':records,'pins_unchanged':True,'accepted_native_run':False,'partial_diagnostic_only':True})
assert all(q['returncode']==0 for q in records)
