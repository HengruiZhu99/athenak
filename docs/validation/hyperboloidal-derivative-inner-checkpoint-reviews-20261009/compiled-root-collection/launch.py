"""One isolated collector invocation; existing destination is never reused."""
import hashlib,json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
def pin(p):
 p=Path(p).resolve();return dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def load(p):return json.loads(Path(p).read_text())
def check(x):assert pin(x['source'])=={k:x[k] for k in ('source','sha256','bytes')}
I=P/'invocation001';I.mkdir(exist_ok=False)
A=P/'authorization.json';a=load(A);check(a['root_review']);review=load(P/'review.json')
for x in review['verified_inputs']:check(x)
S=R/'build-layer-research/continuum/inner-joint-principal-compiled-collector-held-20261009'
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0')
cmd=['/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9','-I','-B',str(S/'collect_once.py'),'--authorization',str(A),'--authorization-sha256',pin(A)['sha256']]
start=time.monotonic()
with (I/'stdout.log').open('wb') as out,(I/'stderr.log').open('wb') as err:
 result=subprocess.run(cmd,cwd=R,env=env,stdout=out,stderr=err,check=False)
drift=[]
for x in review['verified_inputs']:
 try:check(x)
 except BaseException as e:drift.append(dict(source=x['source'],error=repr(e)))
child=load(S/'invocation001/receipt.json') if (S/'invocation001/receipt.json').exists() else None
ok=result.returncode==0 and child is not None and child['completed'] is True and not drift and (I/'stderr.log').stat().st_size==0
r=dict(completed=True,accepted=ok,returncode=result.returncode,seconds=time.monotonic()-start,command=cmd,environment_overrides=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0'),input_drift=drift,stdout=pin(I/'stdout.log'),stderr=pin(I/'stderr.log'),child_receipt=pin(S/'invocation001/receipt.json') if child else None,authorization=pin(A))
(I/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if ok else 1)
