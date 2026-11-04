import pathlib,time,json,subprocess
root=pathlib.Path('build-performance').resolve();start=time.monotonic()
while not (root/'cpu-bench/completed.json').exists():
 if time.monotonic()-start>600:raise RuntimeError('baseline benchmark did not complete; no overlapping run started')
 time.sleep(2)
cmd=['python3','tst/hyperbolic_shift_3d/performance/build_variant.py','--build',str(root/'control'),'--old-source',str(root/'control-source'),'--source',str(root/'m4-source'),'--output',str(root/'m4-bin'),'--experiment','setup']
with (root/'evidence/m4-build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/'evidence/m4-contract.log').open('w') as log:subprocess.run(['mpiexec','-n','1',str(root/'m4-bin/hyperbolic_shift_3d_amr_setup_contract')],stdout=log,stderr=subprocess.STDOUT,timeout=420,check=True)
runs=[]
for policy,p,steps in [('dynamic',2,6),('fixed',4,8)]:
 for repeat in range(3):
  for v in (['m0','m4'] if repeat%2==0 else ['m4','m0']):runs.append({'name':f'setup-{policy}-p{p}-r{repeat}-{v}','variant':v,'ranks':p,'kind':'wave','steps':steps,'policy':policy,'equivalence':f'{policy}-{steps}'})
plan={'variants':{'m0':str(root/'frozen/m0-v2.bin'),'m4':str(root/'m4-bin/hyperbolic_shift_3d_amr_setup')},'runs':runs}
(root/'cpu-m4-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run(['python3','tst/hyperbolic_shift_3d/performance/run_plan.py',str(root/'cpu-m4-plan.json'),str(root/'cpu-m4-bench')],check=True)
