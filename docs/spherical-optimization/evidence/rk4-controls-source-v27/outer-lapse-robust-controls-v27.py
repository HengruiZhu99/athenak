import concurrent.futures,json,subprocess
from pathlib import Path
root=Path.cwd();e=root/'docs/spherical-optimization/evidence';base=json.loads((e/'outer-lapse025-n256-none-40M-v26/manifest.json').read_text())['config']
jobs=[(.55,'none',1e-5,40),(.55,'gauge',1e-5,40),(.55,'constraint',1e-5,40),(.4,'constraint',1e-5,40),(.4,'gauge',2e-5,40),(.6,'none',1e-5,10)]
def run(job):
 c,kind,amp,duration=job;name=f'outer-lapse025-n256-{kind}-{duration}M-c{c:g}-amp{amp:g}-v27';cfg=dict(base,fixed_dt=base['fixed_dt']*c/.4,perturbation=kind,perturb_amplitude=amp,duration=duration,output=str(e/name))
 cmd=[str(root/'build-spherical/python/bin/python'),'tst/spherical_optimization/run.py']
 for key,value in cfg.items():
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:cmd.append(flag)
  elif value is not None:cmd.extend([flag,str(value)])
 with (e/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(job=job,returncode=p.returncode)),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
