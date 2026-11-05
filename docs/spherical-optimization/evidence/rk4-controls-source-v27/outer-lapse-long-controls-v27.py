import concurrent.futures,json,subprocess
from pathlib import Path
root=Path.cwd();e=root/'docs/spherical-optimization/evidence';base=json.loads((e/'outer-lapse025-n256-none-40M-v26/manifest.json').read_text())['config']
jobs=[(xi,div,kind) for xi,div in [(1.,1),(.25,2)] for kind in ['none','gauge']]
def run(job):
 xi,div,kind=job;name=f'outer-lapse{str(xi).replace(".","")}-n256-{kind}-40M-dt{div}-v27';cfg=dict(base,outer_lapse_damping=xi,fixed_dt=base['fixed_dt']/div,perturbation=kind,output=str(e/name))
 cmd=[str(root/'build-spherical/python/bin/python'),'tst/spherical_optimization/run.py']
 for key,value in cfg.items():
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:cmd.append(flag)
  elif value is not None:cmd.extend([flag,str(value)])
 with (e/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(job=job,returncode=p.returncode)),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
