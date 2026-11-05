import concurrent.futures,json,subprocess
from pathlib import Path
root=Path.cwd();e=root/'docs/spherical-optimization/evidence';base=json.loads((e/'outer-lapse025-n128-10M-v26/manifest.json').read_text())['config']
jobs=[(n,kind) for n in [128,256] for kind in ['none','gauge','constraint']]
def run(job):
 n,kind=job;cfg=dict(base,n=n,duration=20.,perturbation=kind,probes=True,fixed_dt=base['fixed_dt']*127.5/(n-.5),output=str(e/f'outer-lapse025-n{n}-{kind}-20M-v26'))
 cmd=[str(root/'build-spherical/python/bin/python'),'tst/spherical_optimization/run.py']
 for key,value in cfg.items():
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:cmd.append(flag)
  elif value is not None:cmd.extend([flag,str(value)])
 with Path(cfg['output']+'.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(job=job,returncode=p.returncode)),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
