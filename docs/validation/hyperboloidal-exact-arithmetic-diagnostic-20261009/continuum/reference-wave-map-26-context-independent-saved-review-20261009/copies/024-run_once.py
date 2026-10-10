"""HELD one-shot local diagnostic build/query/stdlib oracle; exact root release required."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def verify(pins):
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError('protected input changed: '+path)
def metadata(path):return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authorization',type=Path,required=True);ap.add_argument('--authorization-sha256',required=True);args=ap.parse_args()
    r=load(HERE/'recipe.json');out=Path(r['attempt']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();receipt={'completed':False,'returncode':1,'scope':'26-row observational local arithmetic diagnostic only','original_far_Release_passed':False,'commands':[]}
    pins={str(HERE/'recipe.json'):sha(HERE/'recipe.json'),str(Path(__file__).resolve()):sha(__file__)}
    try:
        if not(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize):raise RuntimeError('require -I -B unoptimized')
        if sha(args.authorization)!=args.authorization_sha256:raise RuntimeError('authorization hash mismatch')
        auth=load(args.authorization)
        if not(auth.get('metric_flux_diagnostic_local_admitted') is True and auth['recipe_sha256']==sha(HERE/'recipe.json') and auth['source_index_sha256']==sha(HERE/'source-index.json') and auth['runner_sha256']==sha(__file__)):raise RuntimeError('exact root local release missing')
        if str(Path(sys.executable).resolve())!=r['python']:raise RuntimeError('interpreter mismatch')
        pins.update(load(HERE/'input-pins.json'))
        for row in load(HERE/'source-index.json')['files']:pins[row['path']]=row['sha256']
        pins[str(HERE/'source-index.json')]=auth['source_index_sha256'];pins[str(args.authorization.resolve())]=args.authorization_sha256
        verify(pins);write(out/'pins-before.json',pins)
        old=load(r['original_child_receipt'])
        if not(old['completed'] is False and old['returncode']==1 and old['passed'] is False and old['source_inputs_unchanged'] is True):raise RuntimeError('exact original actual FAIL required')
        env=dict(os.environ);env.update(r['environment'])
        for key in ('PYTHONHOME','PYTHONPATH','PYTHONWARNINGS'):env.pop(key,None)
        exe=out/'metric-flux-probe-release';dep=out/'dependencies.d'
        command=[r['compiler'],*r['compile_flags'],'-MD','-MF',str(dep),str(HERE/'probe.cpp'),'-o',str(exe)]
        jobs=[('compiler-version',[r['compiler'],'--version']),('compile',command),('diagnostic',[str(exe)]),('oracle',[r['python'],'-I','-B',str(HERE/'oracle.py'),'--authorization',str(args.authorization.resolve()),'--attempt',str(out)])]
        for name,cmd in jobs:
            before=time.monotonic()
            stdout=out/('diagnostic.jsonl' if name=='diagnostic' else name+'.stdout');stderr=out/(name+'.stderr')
            with stdout.open('wb') as so,stderr.open('wb') as se:
                child=subprocess.run(cmd,cwd=r['repository'],env=env,stdout=so,stderr=se,check=False)
            receipt['commands'].append({'name':name,'command':cmd,'returncode':child.returncode,'seconds':time.monotonic()-before,'stdout':metadata(stdout),'stderr':metadata(stderr),'environment':{k:env.get(k) for k in r['environment']}})
            if child.returncode:raise RuntimeError('diagnostic stage failed: '+name)
            if name=='compile':
                pins[str(exe)]=sha(exe);pins[str(dep)]=sha(dep)
                text=dep.read_text().replace('\\\n',' ')
                import shlex
                dependencies=[]
                for token in shlex.split(text.split(':',1)[1]):
                    path=Path(token);path=path if path.is_absolute() else Path(r['repository'])/path;path=path.resolve()
                    dependencies.append(metadata(path));pins[str(path)]=sha(path)
                write(out/'dependencies.json',dependencies);receipt['executable']=metadata(exe)
                write(out/'pins-before-query.json',pins)
        result=load(out/'oracle-report.json')
        if not(result['diagnostic_completed'] and result['inputs_unchanged'] and result['rows']==26 and result['original_far_Release_passed'] is False):raise RuntimeError('incomplete observational oracle')
        receipt.update(completed=True,returncode=0,diagnostic_completed=True)
    except BaseException as exc:
        receipt['failure']=type(exc).__name__+': '+str(exc);(out/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:verify(pins);receipt['inputs_unchanged']=True
        except BaseException as exc:receipt.update(returncode=1,inputs_unchanged=False,post_pin_failure=str(exc))
        receipt['seconds']=time.monotonic()-started
        receipt['outputs']=[metadata(path) for path in sorted(out.iterdir()) if path.is_file()]
        write(out/'receipt.json',receipt)
    print(json.dumps(receipt))
    if not(receipt['completed'] and receipt['inputs_unchanged'] and receipt['returncode']==0):raise SystemExit(1)

if __name__=='__main__':main()
