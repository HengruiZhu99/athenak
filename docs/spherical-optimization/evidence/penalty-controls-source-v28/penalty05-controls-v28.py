import concurrent.futures,json,subprocess
from pathlib import Path
root=Path.cwd();e=root/'docs/spherical-optimization/evidence'
jobs=[(256,c,kind,1e-5,40) for c in [.4,.68] for kind in ['none','gauge','constraint']]+[(256,.68,'gauge',2e-5,40)]+[(128,.68,kind,1e-5,40) for kind in ['none','gauge']]+[(512,.68,'none',1e-5,10)]
def run(job):
 n,c,kind,amp,duration=job;ref=e/(f'outer-lapse025-penalty05-n{n}-10M-v28' if n!=128 else 'outer-lapse025-penalty05-n128-10M-v27');cfg=json.loads((ref/'manifest.json').read_text())['config'];name=f'penalty05-n{n}-{kind}-{duration}M-c{c:g}-amp{amp:g}-v28';cfg=dict(cfg,duration=duration,perturbation=kind,perturb_amplitude=amp,probes=duration==40,fixed_dt=cfg['fixed_dt']*c/.4,output=str(e/name));cmd=[str(root/'build-spherical/python/bin/python'),'tst/spherical_optimization/run.py']
 for key,value in cfg.items():
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:cmd.append(flag)
  elif value is not None:cmd.extend([flag,str(value)])
 with (e/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 print(job,p.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
