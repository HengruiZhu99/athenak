#!/usr/bin/env python3
"""Three admitted resolutions on distinct GPU tiles inside one PBS allocation."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);a=p.parse_args();proof=json.loads(a.admission.read_text());source=Path(__file__).resolve().parents[2]
if not proof['passed'] or proof['stage'] not in ['schwarzschild','kerr'] or len(proof['gpu_cases'])!=3:raise RuntimeError('Matching full three-resolution CPU admission required')
for file,digest in proof['delivery_source_sha256'].items():
 if sha(source/'src/test_scalar'/file)!=digest:raise RuntimeError('Scalar source differs from CPU admission')
for file,digest in proof['dependency_source_sha256'].items():
 if sha(source/file)!=digest:raise RuntimeError('Cartesian/shell source differs from CPU admission')
deadline=time.monotonic()+min(1500,(int(os.environ.get('SCALAR_ALLOCATION_WALL_SECONDS','1800'))-300)-(time.time()-float(os.environ['SCALAR_JOB_STARTED'])));tasks=[];records=[];binary=sha(a.exe)
def save(reason=None):
 (a.output/'gpu-receipt.json').write_text(json.dumps(dict(stage=proof['stage'],scientific_admission_sha256=sha(a.admission),records=records,stop_reason=reason,execution='Three independent single-rank resolutions on GPU tiles0.0/1.0/2.0 within this one PBS allocation.'),indent=2)+'\n')
save()
for rank,case in enumerate(proof['gpu_cases']):
 budget=deadline-time.monotonic()-120
 if budget<60:save(dict(kind='budget_before_launch',n=case['n']));break
 if '--output' in case['arguments'] or '--restart' in case['arguments']:raise RuntimeError('GPU output/restart must be controlled explicitly')
 out=a.output/f'{proof["stage"]}-n{case["n"]}';out.mkdir(exist_ok=False);args=[str(a.exe.resolve()),*case['arguments'],'--wall-seconds',str(budget),'--output',str(out.resolve())];command=['mpiexec','-n','1','-ppn','1',f'--cpu-bind=list:{1+8*rank}-{8+8*rank}',*args];environment={**os.environ,'ZE_AFFINITY_MASK':f'{rank}.0','OMP_NUM_THREADS':'1'}
 invocation=dict(binary=binary,arguments=args[1:-2],command=command,environment={k:environment.get(k) for k in ['ONEAPI_DEVICE_SELECTOR','ZE_AFFINITY_MASK','ZE_ENABLE_PCI_ID_DEVICE_ORDER','OMP_NUM_THREADS']},kind='gpu');(out/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n');log=(out/'run.log').open('w');start=time.monotonic();process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment);tasks.append((case,out,command,log,start,process))
for case,out,command,log,start,process in tasks:
 try:returncode=process.wait(timeout=max(1,deadline-time.monotonic()))
 except subprocess.TimeoutExpired:
  process.kill();process.wait();log.close();raise RuntimeError('GPU native deadline exceeded; unknown prefix preserved for inspection')
 log.close();files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};native=dict(returncode=returncode,complete=(out/'complete.txt').exists(),checkpointed=(out/'checkpointed.txt').exists(),elapsed_seconds=time.monotonic()-start,files=files,kind='gpu');(out/'receipt.json').write_text(json.dumps(native,indent=2)+'\n');files['receipt.json']=sha(out/'receipt.json');records.append(dict(n=case['n'],command=command,returncode=returncode,complete=native['complete'],checkpointed=native['checkpointed'],elapsed_seconds=native['elapsed_seconds'],files=files,device_affinity=json.loads((out/'invocation.json').read_text())['environment']['ZE_AFFINITY_MASK']));save()
if any(r['returncode'] for r in records):raise RuntimeError('A GPU scalar resolution failed; all native attempts preserved')
