#!/usr/bin/env python3
"""Run the next admitted scalar CPU stage with bounded, immutable continuations."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from run_cpu import run

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['schwarzschild','kerr'],required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--resolutions',nargs='+',type=int,default=[24,32,40]);p.add_argument('--wall-seconds',type=float,default=2700);p.add_argument('--reuse',nargs='*',type=Path,default=[]);a=p.parse_args()
 admission=json.loads(a.admission.read_text());previous='minkowski' if a.stage=='schwarzschild' else 'schwarzschild'
 a.root.mkdir(parents=True,exist_ok=True)
 if not admission.get('passed') or admission['stage']!=previous:raise RuntimeError('Previous scalar CPU stage is not admitted')
 if admission['equivalent_delivery_binary_sha256']!=sha(a.exe):raise RuntimeError('Executable is not the preserved admitted operator')
 numerical=admission.get('numerical_arguments')
 if not numerical:raise RuntimeError('Exact admitted numerical arguments are required')
 reuse={json.loads((d/'parameters.json').read_text())['n']:d for d in a.reuse}
 if len(reuse)!=len(a.reuse):raise RuntimeError('Duplicate resolution in reuse inputs')
 matrix=[]
 for n in a.resolutions:
  label=f'{a.stage}-n{n}-normal';args=['--n',str(n),'--tlim','100','--mass','1','--spin','0' if a.stage=='schwarzschild' else '.5',*numerical,'--excision','.126' if a.stage=='schwarzschild' else '0','--boundary','radiation','--data','ringdown','--wall-seconds',str(a.wall_seconds)]
  if n in reuse:
   current=reuse[n];meta=json.loads((current/'parameters.json').read_text());receipt=json.loads((current/'receipt.json').read_text());invocation=json.loads((current/'invocation.json').read_text())
   if not (current/'complete.txt').exists() or receipt['returncode'] or invocation['binary'] not in [admission['cpu_binary_sha256'],admission['equivalent_delivery_binary_sha256']]:raise RuntimeError('Reuse requires a completed source-equivalent admitted binary')
   translation={'--normal-upwind':'normal_upwind','--outer-sbp':'outer_sbp','--ko':'ko','--reduction-damping':'reduction_damping','--smooth-layer':'smooth_physical_layer','--radial-stretch':'radial_stretch','--cfl':'cfl'}
   expected=dict(n=n,M=1,a_over_M=0 if a.stage=='schwarzschild' else .5,tlim=100,cap=1,excision=.126 if a.stage=='schwarzschild' else 0,boundary='radiation',data='ringdown')
   for opt,value in zip(numerical[::2],numerical[1::2]):expected[translation[opt]]=float(value)
   if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError('Reused case has different physical or numerical settings')
   if any(not (current/name).exists() or sha(current/name)!=digest for name,digest in receipt['files'].items()):raise RuntimeError('Reused files differ from their immutable receipt')
   if not json.loads((current/'frozen-integrity.json').read_text())['unchanged']:raise RuntimeError('Reused frozen geometry integrity failed')
   print('REUSE',current,flush=True)
  else:current=run(a.exe,a.root,label,args)
  lineage=[str(current.resolve())];index=0
  while not (current/'complete.txt').exists():
   if not (current/'checkpointed.txt').exists():raise RuntimeError('Missing completion/checkpoint status; inspect the existing attempt')
   index+=1;child=a.root/f'{label}-continuation{index}'
   if child.exists():
    record=json.loads((child/'restart-receipt.json').read_text())
    if record['returncode'] or record['checkpoint_sha256']!=sha(current/'checkpoint.f64') or record['binary_sha256']!=sha(a.exe):raise RuntimeError('Existing continuation is failed or mismatched; preserve it')
    print('REUSE',child,flush=True)
   else:
    print('CONTINUE',current,'to',child,flush=True);subprocess.run([sys.executable,str(Path(__file__).with_name('resume_cpu.py')),'--exe',str(a.exe),'--parent',str(current),'--output',str(child),'--tlim','100','--wall-seconds',str(a.wall_seconds)],check=True)
   current=child;lineage.append(str(current.resolve()))
  matrix.append(dict(n=n,final=str(current.resolve()),lineage=lineage,executable_sha256=sha(a.exe)))
  receipt=a.root/f'{label}-lineage.json';text=json.dumps(matrix[-1],indent=2)+'\n'
  if receipt.exists() and receipt.read_text()!=text:raise RuntimeError('Existing lineage differs; preserve both attempts')
  if not receipt.exists():receipt.write_text(text)
 result=a.root/f'{a.stage}-matrix-lineage.json';text=json.dumps(dict(stage=a.stage,cases=matrix,previous_admission_sha256=sha(a.admission)),indent=2)+'\n'
 if result.exists() and result.read_text()!=text:raise RuntimeError('Existing matrix differs')
 if not result.exists():result.write_text(text)
 print(text)
if __name__=='__main__':main()
