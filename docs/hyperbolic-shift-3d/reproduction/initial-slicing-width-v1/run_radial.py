"""Prespecified width sweep; t=0 only, fixed mass and fixed radial norm region."""
from pathlib import Path
import subprocess,concurrent.futures,json,hashlib,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
exe=ROOT/'build-native-initial-smoothing-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_initial_smoothing'
# Keep the inner areal endpoint at 4M; vary only the outer endpoint.
cases=[8,10,12,16,24]
def run(end,q=512,tol='2e-14'):
 stem=f'areal4-{end}-q{q}'+('-tight' if tol!='2e-14' else '')
 cmd=[str(exe),'4',str(end),str(q),tol];start=time.monotonic()
 with (P/(stem+'.csv')).open('x') as out,(P/(stem+'.log')).open('x') as err:
  subprocess.run(cmd,stdout=out,stderr=err,check=True)
 return dict(command=cmd,output=stem+'.csv',seconds=time.monotonic()-start)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  results=list(pool.map(run,cases))
 (P/'radial-commands.json').write_text(json.dumps(results,indent=2)+'\n')
 print(results)
