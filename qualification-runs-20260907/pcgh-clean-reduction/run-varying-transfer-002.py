from pathlib import Path
import subprocess,sys,json,time
root=Path('/Users/hz0693/research/pcgh-clean-reduction-tests-20260907')
repo=Path('/Users/hz0693/research/athenak-pcgh-clean-reduction-20260907')
output=root/'varying-transfer-002';output.mkdir(exist_ok=False)
records=[]
for n in [8,16,32]:
 for smr in [False,True]:
  target=output/f'n{n}-{"smr" if smr else "uniform"}'
  command=[str(root/'venv/bin/python'),str(repo/'analysis/pc_gh_clean_reduction/run_transfer_mesh.py'),'--binary',str(root/'build-current/src/athena'),'--output',str(target),'--profile','smooth','--orders','6','--block-n',str(n)]
  if smr:command+=['--smr']
  start=time.time()
  with (output/f'{target.name}.log').open('w') as log:r=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
  records.append({'command':command,'returncode':r.returncode,'wall_seconds':time.time()-start})
  (output/'controller.json').write_text(json.dumps(records,indent=2)+'\n')
  print(target.name,r.returncode,flush=True)
  if r.returncode:sys.exit(r.returncode)
