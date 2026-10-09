"""Exact one-shot independent saved-field captures for two stopped N24 cases."""
from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=P/'observe_fields.py'
before=json.loads((P/'source-before.json').read_text())
assert sha(source)==before['source_sha256']
assert sha(P/'independent_parser.py')==before['parser_sha256']
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'build-layer-research/boundary/python-deps'))
for tag,name in [('half','wave-map-half-N24-large-t2'),('small','wave-map-N24-small-t2')]:
    recipe=ROOT/('build-layer-research/reference-wave-map-partial-'+tag+'-N24-held-20261009/recipe.json')
    protected=json.loads(recipe.read_text())['fixed_pins']
    protected.update({str(recipe):sha(recipe),str(source):sha(source),str(P/'independent_parser.py'):sha(P/'independent_parser.py'),str(Path(__file__).resolve()):sha(__file__)})
    for f,h in protected.items():assert sha(f)==h,f
    d=P/('invocation-'+tag+'001');d.mkdir(exist_ok=False)
    (d/'pins-before.json').write_text(json.dumps(protected,indent=2)+'\n')
    command=['/Library/Developer/CommandLineTools/usr/bin/python3','-B',str(source),name]
    record={'command':command,'cwd':str(ROOT),'scope':before['scope'],'environment':{k:env[k] for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH']},'source_sha256':sha(source)}
    (d/'command-before.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    with (d/'stdout').open('wb') as so,(d/'stderr').open('wb') as se:
        r=subprocess.run(command,cwd=ROOT,env=env,stdout=so,stderr=se)
    record.update(returncode=r.returncode,seconds=time.monotonic()-start,stdout_sha256=sha(d/'stdout'),stderr_sha256=sha(d/'stderr'))
    try:
        for f,h in protected.items():assert sha(f)==h,f
        record['inputs_unchanged']=True
        (d/'pins-after.json').write_text(json.dumps(protected,indent=2)+'\n')
    except Exception as e:
        record.update(inputs_unchanged=False,pin_error=repr(e))
    (d/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if r.returncode or not record['inputs_unchanged']:raise SystemExit(1)
