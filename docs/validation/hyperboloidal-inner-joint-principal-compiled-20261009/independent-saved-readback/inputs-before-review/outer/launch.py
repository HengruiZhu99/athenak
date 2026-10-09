"""Single fresh isolated invocation of the root-reviewed compiled gate."""
import hashlib,json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
def pin(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def load(p):return json.loads(Path(p).read_text())
def check(x):assert pin(x['path'])=={k:x[k] for k in ('path','sha256','bytes')},x['path']
I=P/'invocation001';I.mkdir(exist_ok=False)
review=load(P/'review.json');auth=load(P/'authorization.json')
check(auth['root_review'])
for x in review['verified_inputs']:check(x)
S=Path(review['source_index']['path']).parent
recipe=load(S/'recipe.json')
env=os.environ.copy();env.update(PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
cmd=[recipe['python']['path'],'-I','-B',str(S/'run_gate.py'),'--authorization',str(P/'authorization.json')]
start=time.monotonic()
with (I/'stdout.log').open('wb') as out,(I/'stderr.log').open('wb') as err:
 result=subprocess.run(cmd,cwd=R,env=env,stdout=out,stderr=err,check=False)
drift=[]
for x in review['verified_inputs']:
 try:check(x)
 except BaseException as e:drift.append(dict(path=x['path'],error=repr(e)))
child=load(S/'attempts/gate001/receipt.json') if (S/'attempts/gate001/receipt.json').exists() else None
ok=result.returncode==0 and child is not None and child['passed'] is True and child['inputs_unchanged'] is True and not drift and (I/'stderr.log').stat().st_size==0
receipt=dict(completed=True,accepted=ok,command=cmd,environment_overrides={k:env[k] for k in ('PYTHONOPTIMIZE','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},returncode=result.returncode,seconds=time.monotonic()-start,source_drift=drift,stdout=pin(I/'stdout.log'),stderr=pin(I/'stderr.log'),child_receipt=pin(S/'attempts/gate001/receipt.json') if child else None,authorization=pin(P/'authorization.json'))
(I/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(receipt,indent=2));raise SystemExit(0 if ok else 1)
