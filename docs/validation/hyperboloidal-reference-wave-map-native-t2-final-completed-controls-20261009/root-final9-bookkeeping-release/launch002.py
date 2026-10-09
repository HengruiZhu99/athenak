from pathlib import Path
import hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent
SUITE=HERE.parent/'reference-wave-map-t2-subset-comparison-held-v3-20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inv=HERE/'invocation002';inv.mkdir(exist_ok=False)
auth=HERE/'authorization.json';source=SUITE/'run_subset.py';pins={str(p):sha(p) for p in [auth,source,SUITE/'compare_subset.py',SUITE/'source-index.json',Path(__file__).resolve()]}
cmd=['/Library/Developer/CommandLineTools/usr/bin/python3','-B',str(source),str(auth)]
(inv/'before.json').write_text(json.dumps({'command':cmd,'pins':pins},indent=2)+'\n');env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';started=time.monotonic()
with (inv/'stdout.log').open('wb') as out,(inv/'stderr.log').open('wb') as err:r=subprocess.run(cmd,cwd=HERE.parents[1],env=env,stdout=out,stderr=err)
receipt={'returncode':r.returncode,'seconds':time.monotonic()-started,'sources_unchanged':all(sha(Path(p))==h for p,h in pins.items()),'stdout_sha256':sha(inv/'stdout.log'),'stderr_sha256':sha(inv/'stderr.log')};(inv/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(r.returncode)
