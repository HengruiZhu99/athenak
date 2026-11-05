#!/usr/bin/env python3
"""Explicit experimental CFL sweep; retains every subprocess and failed run."""
import argparse,json,subprocess,sys
from pathlib import Path
from kernel import Kernel
from evolution import ChartEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--n',type=int,default=64);ap.add_argument('--duration',type=float,default=1);ap.add_argument('--targets',type=float,nargs='+',default=[.1,.2,.3,.4,.5,.6,.7]);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 k=Kernel(a.library,separated=True);e=ChartEvolution(k,a.n,normal_policy='analytic',reference_jets=True,reference_step=.000125);e.rhs(e.initial);speed=e.last['speed_over_dr'];k.close();rows=[]
 for cfl in a.targets:
  folder=a.output/f'C{cfl:.3f}';command=[sys.executable,str(Path(__file__).with_name('run.py')),'--library',a.library,'--output',str(folder),'--n',str(a.n),'--duration',str(a.duration),'--separated','--reference-jets','--reference-step','.000125','--normal-policy','analytic','--algebraic-chart','--experimental-timestep','--fixed-dt',repr(cfl/speed)]
  process=subprocess.run(command,capture_output=True,text=True)
  (a.output/f'C{cfl:.3f}.log').write_text(process.stdout+process.stderr)
  result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
  rows.append(dict(C_char_target=cfl,command=command,returncode=process.returncode,result=result))
  (a.output/'sweep.json').write_text(json.dumps(dict(n=a.n,duration_M=a.duration,initial_speed_over_dr=speed,runs=rows,qualification='Explicit short-run exploration above default guards; no robust or long-duration timestep admission.'),indent=2)+'\n')
  print(cfl,result['status'] if result else 'process_failed',flush=True)
if __name__=='__main__':main()
