from pathlib import Path
import json,hashlib,subprocess,os,time,traceback
P=Path(__file__).resolve().parent
S=P.parent/'continuum/native-angular-pulse-flat-derivatives-timing-launch-held-20261009/run_derivatives_once.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inv=P/'invocation001';inv.mkdir(exist_ok=False);auth=P/'authorization.json';pins={str(p):sha(p) for p in [S,auth,Path(__file__).resolve()]};cmd=['/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python','-B',str(S),'--authorization',str(auth)];env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');(inv/'before.json').write_text(json.dumps({'command':cmd,'pins':pins},indent=2)+'\n');start=time.monotonic();receipt={'returncode':None}
try:
 with (inv/'stdout.log').open('wb') as out,(inv/'stderr.log').open('wb') as err:receipt['returncode']=subprocess.run(cmd,cwd=S.parent,env=env,stdout=out,stderr=err).returncode
except BaseException as e:receipt.update(error=repr(e),traceback=traceback.format_exc())
finally:
 receipt.update(seconds=time.monotonic()-start,sources_unchanged=all(sha(Path(p))==h for p,h in pins.items()));(inv/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));raise SystemExit(0 if receipt['returncode']==0 and receipt['sources_unchanged'] else 1)
