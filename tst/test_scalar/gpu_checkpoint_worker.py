#!/usr/bin/env python3
"""Run only missing CPU-admitted GPU prefixes with the original GPU executable."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from run_cpu import run

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());proof=json.loads(a.admission.read_text());source=Path(__file__).resolve().parents[2]
if not proof['passed'] or proof['stage'] not in ['schwarzschild','kerr'] or plan['kind']!='gpu_validation' or sha(a.exe)!=plan['binary_sha256'] or sha(a.admission)!=plan['cpu_admission_sha256']:raise RuntimeError('Exact preserved GPU executable and full CPU scientific admission required')
for file,digest in proof['delivery_source_sha256'].items():
 if sha(source/'src/test_scalar'/file)!=digest:raise RuntimeError('GPU scalar source differs from CPU admission')
for file,digest in proof['dependency_source_sha256'].items():
 if sha(source/file)!=digest:raise RuntimeError('GPU Cartesian/shell dependency differs from CPU admission')
os.environ['SCALAR_CPU_THREADS']='1';os.environ['SCALAR_CPU_PROC_BIND']='false';deadline=float(os.environ['SCALAR_JOB_STARTED'])+(int(os.environ.get('SCALAR_ALLOCATION_WALL_SECONDS','1800'))-300);records=[]
def save():
 (a.output/'gpu-checkpoint-worker-receipt.json').write_text(json.dumps(dict(records=records,plan_sha256=sha(a.plan),binary_sha256=sha(a.exe),cpu_admission_sha256=sha(a.admission),all_cases_complete=len(records)==len(plan['cases']) and all(r['complete'] for r in records)),indent=2)+'\n')
save()
for case in plan['cases']:
 budget=deadline-time.time()-120
 if budget<120:break
 admitted=next(c for c in proof['gpu_cases'] if c['n']==case['n'])
 if case['admitted_arguments']!=admitted['arguments']:raise RuntimeError('Pending GPU case differs from the exact admitted matrix')
 out=a.output/case['label']
 if case['action']=='resume':
  parent=Path(case['parent']);receipt=parent/('restart-receipt.json' if (parent/'restart-receipt.json').exists() else 'receipt.json');r=json.loads(receipt.read_text())
  if (parent/'complete.txt').exists() or not (parent/'checkpointed.txt').exists() or r['returncode']!=0:raise RuntimeError('Only a confirmed incomplete GPU wall checkpoint may be resumed')
  if any(not (parent/f).is_file() or sha(parent/f)!=h for f,h in r['files'].items()):raise RuntimeError('Native GPU parent payload/metadata hash mismatch')
  params=json.loads((parent/'parameters.json').read_text());expected=dict(L=3,S=24,C=10,compatible_reduction=False,characteristic_transfer=False,reference_layer='original_quintic',stored_auxiliary='normal_Pi',halo_width=4)
  keys={'--n':'n','--mass':'M','--spin':'a_over_M','--cap':'cap','--excision':'excision','--boundary':'boundary','--data':'data','--tlim':'tlim','--normal-upwind':'normal_upwind','--outer-sbp':'outer_sbp','--ko':'ko','--reduction-damping':'reduction_damping','--smooth-layer':'smooth_physical_layer','--radial-stretch':'radial_stretch','--cfl':'cfl'}
  for flag,value in zip(admitted['arguments'][::2],admitted['arguments'][1::2]):expected[keys[flag]]=value if flag in ['--boundary','--data'] else float(value)
  if any(params.get(k)!=v for k,v in expected.items()):raise RuntimeError('GPU checkpoint scalar/geometry settings differ from exact CPU-admitted case')
  subprocess.run([sys.executable,str(Path(__file__).with_name('resume_cpu.py')),'--exe',str(a.exe),'--parent',str(parent),'--output',str(out),'--tlim','100','--wall-seconds',str(budget)],check=True);native=out/'restart-receipt.json'
 elif case['action']=='run':
  out=run(a.exe,a.output,case['label'],admitted['arguments']+['--wall-seconds',str(budget)]);native=out/'receipt.json'
 else:raise RuntimeError('Unknown GPU checkpoint action')
 backend=(out/'backend.txt').read_text()
 if 'Default Device: SYCL' not in backend or 'Is GPU: 1' not in backend or 'Intel' not in backend:raise RuntimeError('Continuation did not execute on the actual Intel SYCL GPU')
 complete=(out/'complete.txt').exists();records.append(dict(n=case['n'],label=case['label'],action=case['action'],complete=complete,checkpointed=(out/'checkpointed.txt').exists(),native_receipt_sha256=sha(native)));save()
 if not complete:break
