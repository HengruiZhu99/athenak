from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent;R=P.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
S=R/'build-layer-research/boundary/reference-wave-map-N24-controls-collector-held-20261009/collect_once.py';A=P/'authorization.json'
assert sha(S)=='f02c8a15884a867a2a66ef9535134fd751fba7528818e20f58f7839198923ef4'
assert sha(A)=='a0cd072de963a00569429af0ef0d26243389a8baf62ccd9ce6793b6f9d3d10ef'
D=P/'invocation001';D.mkdir(exist_ok=False)
command=['/Library/Developer/CommandLineTools/usr/bin/python3','-B',str(S),'--authorization',str(A),'--authorization-sha256',sha(A)]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
before={'command':command,'source_sha256':sha(__file__),'child_source_sha256':sha(S),'authorization_sha256':sha(A),'cwd':str(R)}
(D/'before.json').write_text(json.dumps(before,indent=2)+'\n');start=time.monotonic()
with (D/'stdout').open('wb') as so,(D/'stderr').open('wb') as se:r=subprocess.run(command,cwd=R,env=env,stdout=so,stderr=se)
record={**before,'returncode':r.returncode,'seconds':time.monotonic()-start,'source_unchanged':sha(__file__)==before['source_sha256'],'child_source_unchanged':sha(S)==before['child_source_sha256'],'authorization_unchanged':sha(A)==before['authorization_sha256'],'stdout_sha256':sha(D/'stdout'),'stderr_sha256':sha(D/'stderr')}
(D/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(r.returncode)
