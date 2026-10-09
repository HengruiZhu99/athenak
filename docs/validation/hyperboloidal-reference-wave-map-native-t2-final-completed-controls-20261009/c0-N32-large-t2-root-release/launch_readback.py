"""One-shot exact original completed C0 N32 readback outer capture."""
from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
S=ROOT/'build-layer-research/reference-wave-map-t2-readback-held-20261009/run_t2_readback.py'
A=P/'authorization.json'
EXPECTED='50596409891c60bae9f8a559d482cbef1853e32d0c19fa3b009c65676100982b'
assert sha(A)==EXPECTED
D=P/'invocation001';D.mkdir(exist_ok=False)
cmd=['/Library/Developer/CommandLineTools/usr/bin/python3','-B',str(S),str(A),'c0-N32-large-t2']
overrides={'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'build-layer-research/boundary/python-deps')}
env=dict(os.environ,**overrides)
before={'command':cmd,'environment':overrides,'cwd':str(ROOT),'source_sha256':sha(__file__),'wrapper_sha256':sha(S),'authorization_sha256':sha(A)}
(D/'before.json').write_text(json.dumps(before,indent=2)+'\n')
start=time.monotonic()
with (D/'stdout').open('wb') as so,(D/'stderr').open('wb') as se:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=so,stderr=se)
record={**before,'returncode':r.returncode,'seconds':time.monotonic()-start,'source_unchanged':sha(__file__)==before['source_sha256'],'authorization_unchanged':sha(A)==before['authorization_sha256'],'wrapper_unchanged':sha(S)==before['wrapper_sha256'],'stdout_sha256':sha(D/'stdout'),'stderr_sha256':sha(D/'stderr')}
(D/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));raise SystemExit(r.returncode)
