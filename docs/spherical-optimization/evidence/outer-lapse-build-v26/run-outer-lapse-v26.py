import concurrent.futures,json,subprocess
from pathlib import Path
root=Path.cwd();e=root/'docs/spherical-optimization/evidence';base=json.loads((e/'compatible-lapse-f1-n128-10M-v23/manifest.json').read_text())['config']
def run(n):
 cfg=dict(base,library='build-spherical-v26/libcurrent_spherical.dylib',n=n,outer_lapse_damping=.25,fixed_dt=base['fixed_dt']*127.5/(n-.5),output=str(e/f'outer-lapse025-n{n}-10M-v26'))
 cmd=[str(root/'build-spherical/python/bin/python'),'tst/spherical_optimization/run.py']
 for key,value in cfg.items():
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:cmd.append(flag)
  elif value is not None:cmd.extend([flag,str(value)])
 with (e/f'outer-lapse025-n{n}-10M-v26.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(n=n,returncode=p.returncode)),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[64,128,256,512]))
