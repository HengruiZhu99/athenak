#!/usr/bin/env python3
"""Run an immutable bounded plan; preserve every command, failure and checkpoint.

Performance plans must use alternating same-P pairs. This runner never submits
jobs, changes science, retries failures, or deletes artifacts.
"""
import argparse, hashlib, json, os, pathlib, shutil, signal, subprocess, time

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,data):
    p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan');parser.add_argument('output');a=parser.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',plan);(out/'bin').mkdir();variants={};hashes={}
    for name,path in plan['variants'].items():
        dest=out/'bin'/name;shutil.copy2(path,dest);variants[name]=str(dest);hashes[name]=digest(dest)
    save(out/'binaries.json',hashes)
    if plan.get('aurora'):
        nodes=sorted(set(pathlib.Path(os.environ['PBS_NODEFILE']).read_text().split()))
        assert len(nodes)==plan['nodes'] and 1<=len(nodes)<=2
        assert os.environ['MPIR_CVAR_ENABLE_GPU']=='1'
        save(out/'allocation.json',{'job':os.environ['PBS_JOBID'],'nodes':nodes,'requested_nodes':plan['nodes'],'walltime_limit_seconds':plan['deadline_seconds'],'environment':{k:v for k,v in os.environ.items() if k.startswith(('PBS_','ZE_','ONEAPI_','MPIR_','OMP_','EnableImplicit'))}})
    results=[];comparisons={};began=time.monotonic()
    for i,case in enumerate(plan['runs']):
        if time.monotonic()-began>plan.get('deadline_seconds',100000)-case.get('timeout',420)-120:
            save(out/'stopped.json',{'reason':'insufficient time for next bounded case','next_case':i});return 3
        if shutil.disk_usage(out).free<5*1024**3:raise RuntimeError('insufficient free storage including checkpoint reserve')
        name=case.get('name',f"{i:03d}-{case['variant']}-p{case['ranks']}");prefix=out/name
        ranks=case['ranks'];cmd=['mpiexec','-n',str(ranks)]
        if plan.get('aurora'):
            ppn=case.get('ppn',min(ranks,12));assert ppn<=12 and ranks<=ppn*plan['nodes']
            bind='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
            cmd+=['-ppn',str(ppn),'--cpu-bind=list:'+':'.join(bind.split(':')[:ppn]),'gpu_tile_compact.sh']
        cmd+=[variants[case['variant']],str(case.get('n',12)),case.get('kind','wave'),str(case.get('steps',8)),'.00025',str(prefix),case.get('restart','-'),case.get('policy','fixed')]
        receipt={'case':case,'command':cmd,'binary_sha256':hashes[case['variant']],'status':'started','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        save(str(prefix)+'.receipt.json',receipt);env=os.environ.copy();env.update(case.get('env',{}));start=time.monotonic()
        with open(str(prefix)+'.log','w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=proc.wait(timeout=case.get('timeout',420))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code='timeout'
        receipt.update(status='completed',returncode=code,seconds=time.monotonic()-start)
        for suffix in ['checkpoint','failed-stage.bin','rejected-transfer.bin']:
            p=pathlib.Path(str(prefix)+'.'+suffix)
            if p.exists():receipt[suffix]={'sha256':digest(p),'bytes':p.stat().st_size}
        if digest(variants[case['variant']])!=hashes[case['variant']]:raise RuntimeError('binary changed during run')
        expected=case.get('expected',0);receipt['passed']=code==expected
        if expected==2:receipt['passed']&='AMR invalid interpolated geometry' in pathlib.Path(str(prefix)+'.log').read_text()
        key=case.get('equivalence')
        if key and code==0:
            sha=receipt['checkpoint']['sha256'];prior=comparisons.setdefault(key,sha)
            receipt['bitwise_checkpoint_equal']=sha==prior;receipt['passed']&=sha==prior
        save(str(prefix)+'.receipt.json',receipt);results.append(receipt);save(out/'results.json',results)
        print(json.dumps({'name':name,'seconds':receipt['seconds'],'passed':receipt['passed']}),flush=True)
        if not receipt['passed']:raise RuntimeError('preserved failed case: '+name)
    save(out/'completed.json',{'runs':len(results),'seconds':time.monotonic()-began});return 0
if __name__=='__main__':raise SystemExit(main())
