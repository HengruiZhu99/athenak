#!/usr/bin/env python3
"""Explicit experimental CFL sweep; preserves the selected formulation and failures."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--run',type=Path,help='Copy the complete configuration of an immutable run')
 ap.add_argument('--n',type=int);ap.add_argument('--duration',type=float)
 ap.add_argument('--perturbation',choices=('none','gauge','constraint'))
 ap.add_argument('--targets',type=float,nargs='+',default=[.1,.2,.3,.4,.5,.6,.7]);a=ap.parse_args()
 if any(not np.isfinite(c) or c<=0 for c in a.targets):raise ValueError('CFL targets must be positive and finite')
 if len({f'C{c:.3f}' for c in a.targets})!=len(a.targets):raise ValueError('CFL folder names collide at three decimals')
 cfg=json.loads((a.run/'manifest.json').read_text())['config'] if a.run else dict(n=64,duration=1.,separated=True,normal_policy='analytic',reference_jets=True,reference_step=.000125,algebraic_chart=True)
 cfg=dict(cfg,library=a.library,experimental_timestep=True)
 for key in ('n','duration','perturbation'):
  if getattr(a,key) is not None:cfg[key]=getattr(a,key)
 if cfg['n']<16 or not np.isfinite(cfg['duration']) or cfg['duration']<=0:raise ValueError('invalid grid or duration')
 a.output.mkdir(parents=True,exist_ok=False)
 k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);e.rhs(e.initial);speed=float(e.last['speed_over_dr']);k.close();rows=[]
 identity=dict(n=cfg['n'],duration_M=cfg['duration'],initial_speed_over_dr=speed,configuration=cfg,base_manifest_sha256=hashlib.sha256((a.run/'manifest.json').read_bytes()).hexdigest() if a.run else None,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Explicit nonlinear exploration above default guards; completion alone does not admit a robust timestep. Every candidate uses the same complete formulation configuration.')
 for cfl in a.targets:
  folder=a.output/f'C{cfl:.3f}';trial=dict(cfg,output=str(folder),fixed_dt=cfl/speed)
  command=[sys.executable,str(Path(__file__).with_name('run.py'))]
  for key,value in trial.items():
   flag='--'+key.replace('_','-')
   if isinstance(value,bool):
    if value:command.append(flag)
   elif value is not None:command.extend([flag,str(value)])
  process=subprocess.run(command,capture_output=True,text=True)
  (a.output/f'C{cfl:.3f}.log').write_text(process.stdout+process.stderr)
  result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
  rows.append(dict(C_char_target=cfl,command=command,returncode=process.returncode,result=result))
  (a.output/'sweep.json').write_text(json.dumps(dict(identity,runs=rows),indent=2)+'\n')
  print(cfl,result['status'] if result else 'process_failed',flush=True)
if __name__=='__main__':main()
