#!/usr/bin/env python3
"""Bounded local MPI qualification; preserve logs, failures and checkpoint hashes."""
import argparse, hashlib, json, pathlib, subprocess, time, shutil
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--output',required=True);p.add_argument('--mpiexec',default='mpiexec');p.add_argument('--timeout',type=int,default=600);p.add_argument('--suite',choices=['matrix','boost','restart','controls','equilibrium'],default='matrix');a=p.parse_args()
out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
binary=out/"evolution.bin";shutil.copy2(pathlib.Path(a.binary).resolve(),binary)
binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()
results=[]
def run(name,ranks,kind,steps,restart='-',policy='dynamic',expected=0):
    prefix=out/name
    cmd=[a.mpiexec,'-n',str(ranks),str(binary),'12',kind,str(steps),'.00025',str(prefix),str(restart),policy]
    start=time.monotonic()
    with prefix.with_suffix('.log').open('w') as log:
        try: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout);status=result.returncode
        except subprocess.TimeoutExpired: status='timeout'
    record={'binary_sha256':binary_sha256,'name':name,'command':cmd,'returncode':status,'seconds':time.monotonic()-start}
    for suffix in ['report.json','failure.json','checkpoint']:
        f=pathlib.Path(str(prefix)+'.'+suffix)
        if f.exists():record[suffix]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    results.append(record);(out/'manifest.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)
    if expected==2 and 'AMR invalid interpolated geometry' not in prefix.with_suffix('.log').read_text():raise RuntimeError('unexpected failure mode')
    if status!=expected:raise RuntimeError(f'{name} failed; see preserved log')
    return pathlib.Path(str(prefix)+'.checkpoint')
if a.suite=='matrix':
    for ranks in [1,2,4]:run(f'wave-{ranks}',ranks,'wave',6)
    run('wave-uniform',1,'wave',6,policy='uniform')
    run('minkowski',2,'minkowski',6)
elif a.suite=='boost':
    run('boost',2,'boost',6)
    run('boost-uniform',2,'boost',6,policy='uniform')
elif a.suite=='equilibrium':
    run('minkowski',2,'minkowski',6)
    run('minkowski-uniform',1,'minkowski',6,policy='uniform')
elif a.suite=='controls':
    run('wave-final-2',2,'wave',6)
    run('boost-rejected',2,'boost',6,expected=2)
    run('boost-fixed',1,'boost',6,policy='fixed')
    first=run('boost-save',2,'boost',3,policy='fixed')
    run('boost-restart',4,'boost',6,restart=first,policy='fixed')
    run('boost-uniform',2,'boost',6,policy='uniform')
else:
    run('wave-full',1,'wave',6)
    first=run('wave-save',2,'wave',3)
    run('wave-restart',4,'wave',6,restart=first)
