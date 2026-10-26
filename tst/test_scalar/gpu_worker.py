#!/usr/bin/env python3
"""Run only a CPU-admitted scalar stage in one bounded GPU allocation."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);a=p.parse_args()
admission=json.loads(a.admission.read_text())
if not admission.get('passed') or admission.get('stage') not in ['minkowski','schwarzschild','kerr']:raise RuntimeError('Missing scientific CPU admission')
binding=admission.get('delivery_source_sha256')
if not binding:raise RuntimeError('Missing CPU source binding')
source_root=Path(__file__).resolve().parents[2]/'src/test_scalar'
if any(hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=digest for name,digest in binding.items()):raise RuntimeError('GPU source differs from the CPU-admitted operator')
dependencies=admission.get('dependency_source_sha256')
if not dependencies or any(hashlib.sha256((Path(__file__).resolve().parents[2]/name).read_bytes()).hexdigest()!=digest for name,digest in dependencies.items()):raise RuntimeError('GPU Cartesian/cubed-sphere dependencies differ from CPU admission')
if admission['stage'] in ['schwarzschild','kerr']:
 os.execv(sys.executable,[sys.executable,str(Path(__file__).with_name('gpu_parallel_worker.py')),'--exe',str(a.exe),'--output',str(a.output),'--admission',str(a.admission)])
cases=admission.get('gpu_cases')
if cases is None:
 if admission['stage']!='minkowski' or admission.get('configuration','original')!='original' or admission.get('resolutions',[12,18,24])!=[12,18,24]:raise RuntimeError('Admission must specify the exact CPU-validated scalar arguments')
 cases=[dict(n=n,arguments=['--n',str(n),'--tlim','12']) for n in [12,18,24]]
records=[];remaining=min(1500,(int(os.environ.get('SCALAR_ALLOCATION_WALL_SECONDS','1800'))-300)-(time.time()-float(os.environ.get('SCALAR_JOB_STARTED',time.time()))));deadline=time.monotonic()+remaining
def save(stop_reason=None):
 (a.output/'gpu-receipt.json').write_text(json.dumps(dict(stage=admission['stage'],scientific_admission_sha256=hashlib.sha256(a.admission.read_bytes()).hexdigest(),records=records,stop_reason=stop_reason),indent=2)+'\n')
save()
for case in cases:
 n=case['n'];scalar_args=case['arguments']
 if '--output' in scalar_args or '--restart' in scalar_args:raise RuntimeError('GPU matrix output and restarts must be managed explicitly')
 budget=deadline-time.monotonic()-120
 if budget<60:save(dict(kind='budget_before_launch',n=n));break
 out=a.output/f'{admission["stage"]}-n{n}';out.mkdir(exist_ok=False)
 args=[str(a.exe.resolve()),*scalar_args,'--wall-seconds',str(budget),'--output',str(out.resolve())]
 cmd=['mpiexec','-n','1','-ppn','1','--cpu-bind=list:1-8',*args];start=time.monotonic()
 invocation=dict(binary=hashlib.sha256(a.exe.read_bytes()).hexdigest(),arguments=args[1:-2],environment={k:os.environ.get(k) for k in ['ONEAPI_DEVICE_SELECTOR','ZE_AFFINITY_MASK','OMP_NUM_THREADS']},command=cmd,kind='gpu')
 (out/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
 with (out/'run.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-start))
 files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
 (out/'receipt.json').write_text(json.dumps(dict(returncode=r.returncode,complete=(out/'complete.txt').exists(),checkpointed=(out/'checkpointed.txt').exists(),elapsed_seconds=time.monotonic()-start,files=files,kind='gpu'),indent=2)+'\n')
 files['receipt.json']=hashlib.sha256((out/'receipt.json').read_bytes()).hexdigest()
 records.append(dict(n=n,command=cmd,returncode=r.returncode,complete=(out/'complete.txt').exists(),checkpointed=(out/'checkpointed.txt').exists(),elapsed_seconds=time.monotonic()-start,files=files))
 save()
 if r.returncode:raise RuntimeError('GPU scalar run failed; preserve attempt')
 if not (out/'complete.txt').exists():break # Preserve checkpoint and resume this prefix explicitly in a later allocation.
