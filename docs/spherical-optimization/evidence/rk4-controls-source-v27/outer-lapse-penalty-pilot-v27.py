import concurrent.futures,json,subprocess
from pathlib import Path
root=Path.cwd();e=root/'docs/spherical-optimization/evidence'
def run(job):
 n,duration=job;base=json.loads((e/f'outer-lapse025-n{n}-10M-v26/manifest.json').read_text())['config'];cfg=dict(base,duration=duration,penalty=.5,output=str(e/f'outer-lapse025-penalty05-n{n}-{duration:g}M-v27'));cmd=[str(root/'build-spherical/python/bin/python'),'tst/spherical_optimization/run.py']
 for key,value in cfg.items():
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:cmd.append(flag)
  elif value is not None:cmd.extend([flag,str(value)])
 with Path(cfg['output']+'.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 print(job,p.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[(64,.1),(128,10.)]))
