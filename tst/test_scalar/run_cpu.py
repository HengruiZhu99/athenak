#!/usr/bin/env python3
"""Hash-addressed scalar runs. Existing attempts are immutable; no duplicate launch."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(exe,root,label,args):
 output=root/label;key=dict(binary=sha(exe),arguments=args,environment={'OMP_NUM_THREADS':os.environ.get('SCALAR_CPU_THREADS','6'),'OMP_PROC_BIND':os.environ.get('SCALAR_CPU_PROC_BIND','false')})
 if 'SCALAR_CPU_PLACES' in os.environ:key['environment']['OMP_PLACES']=os.environ['SCALAR_CPU_PLACES']
 key['identity']=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
 if output.exists():
  receipt=output/'invocation.json'
  if receipt.exists() and json.loads(receipt.read_text())['identity']==key['identity']:
   if not (output/'receipt.json').exists():raise RuntimeError('Existing attempt has unknown or active status; inspect it before continuing: '+str(output))
   if json.loads((output/'receipt.json').read_text())['returncode']!=0:raise RuntimeError('Existing failed attempt retained: '+str(output))
   print('REUSE',label,flush=True);return output
  raise RuntimeError('Existing evidence cannot be overwritten: '+str(output))
 output.mkdir(parents=True);(output/'invocation.json').write_text(json.dumps(key,indent=2)+'\n')
 cmd=[str(exe.resolve()),*args,'--output',str(output.resolve())];print('RUN',label,flush=True);start=time.time()
 with (output/'run.log').open('w') as log:
  completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**key['environment']})
 result=dict(returncode=completed.returncode,elapsed_seconds=time.time()-start,complete=(output/'complete.txt').exists(),checkpointed=(output/'checkpointed.txt').exists(),files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
 (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 if completed.returncode:raise RuntimeError('Scalar attempt failed; retained at '+str(output))
 return output

def main():
 p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--matrix',choices=['refinement','time','outer','schwarzschild','kerr'],required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
 if a.matrix=='refinement':cases=[(f'minkowski-n{n}',['--n',str(n),'--tlim','12']) for n in [18,24]]
 elif a.matrix=='time':cases=[(f'minkowski-n18-cfl{cfl}',['--n','18','--tlim','12','--cfl',str(cfl)]) for cfl in [.09,.045]]
 elif a.matrix=='outer':cases=[(f'minkowski-n18-cap{cap}',['--n','18','--tlim','12','--cap',str(cap),'--boundary','radiation']) for cap in [.9,.95]]
 else:
  spin='0' if a.matrix=='schwarzschild' else '.5'
  cases=[(f'{a.matrix}-n{n}',['--n',str(n),'--tlim','100','--mass','1','--spin',spin,'--boundary','radiation','--data','ringdown','--excision','0.126']) for n in [12,18,24]]
 for label,args in cases:run(a.exe,a.root,label,args)
if __name__=='__main__':main()
