"""One-shot exact N32 collector or partial-observer root release capture."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
P=Path(__file__).resolve().parent; R=P.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mode=sys.argv[1];assert mode in ['collector','partial']
A=P/('authorization-'+mode+'.json')
expected={'collector':'1085f88b915333fe4b060dd54485fc10cfe7c4e936198fa344e10362dc560bae','partial':'e8bd17d73dfc22301f2b03ae319ab3ec39d2204b29b70a0eae7926f80510137c'}[mode]
assert sha(A)==expected
if mode=='collector':
 S=R/'build-layer-research/wave-map-native-t2-wave-N32-failure-stage-20261009/collect_failure.py'
 assert sha(S)=='3f7300c1a7b746dcaeb3bf69adeb079e6a2fa7506239ad1a99729604d35ae2f3'
 command=['/Library/Developer/CommandLineTools/usr/bin/python3','-B',str(S)]
else:
 S=R/'build-layer-research/reference-wave-map-partial-N32-held-20261009/run_released_observation001.py'
 assert sha(S)=='9a1edeee0f67917218227766924b8f70197c4b538eec7c10c4e22c4cdba52e2e'
 command=['/Library/Developer/CommandLineTools/usr/bin/python3','-B',str(S),str(A),expected]
D=P/(mode+'-invocation001');D.mkdir(exist_ok=False)
env=os.environ.copy();overrides={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'};env.update(overrides)
before={'command':command,'cwd':str(R),'environment_overrides':overrides,'source_sha256':sha(__file__),'child_source_sha256':sha(S),'authorization_sha256':sha(A),'mode':mode,'scope':'One-shot failed native N32 preservation/partial observation; never completed native acceptance.'}
(D/'before.json').write_text(json.dumps(before,indent=2)+'\n')
start=time.monotonic()
with (D/'stdout').open('wb') as so,(D/'stderr').open('wb') as se:run=subprocess.run(command,cwd=R,env=env,stdout=so,stderr=se)
record={**before,'returncode':run.returncode,'seconds':time.monotonic()-start,'source_unchanged':sha(__file__)==before['source_sha256'],'child_source_unchanged':sha(S)==before['child_source_sha256'],'authorization_unchanged':sha(A)==before['authorization_sha256'],'stdout_sha256':sha(D/'stdout'),'stderr_sha256':sha(D/'stderr')}
(D/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(run.returncode)
