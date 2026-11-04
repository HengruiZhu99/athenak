"""Fresh bounded N12 control: refine .78-.90 radial sampling fourfold."""
from pathlib import Path
import json,hashlib,subprocess,time,shutil
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
B=R/'output/refined-convergence/shell-angular-candidate-v1/attempt02';O=R/'output/refined-convergence/boosted-radial24-control';O.mkdir(exist_ok=False)
E=D/'boosted-radial24-control';E.mkdir(exist_ok=False)
assert json.loads((D/'shell-angular-checks/verification.json').read_text())['passed']
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial24_angular2_reference.cfg'
cfg=O/'uniform-core.cfg';cfg.write_text(config.read_text().replace('smr_levels=3','smr_levels=0').replace('smr_half_widths=0.075,0.025,0.025',''))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
commands=[]
def run(name,cmd):
 start=time.time();cmd=[str(x) for x in cmd]
 with (E/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 commands.append(dict(command=cmd,returncode=p.returncode,elapsed=time.time()-start));(E/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert p.returncode==0
 print(name,'completed',flush=True)
run('configuration',['mpiexec','-n',1,B/'support',config,O/'unused.json','config-only'])
prefix=O/'state';run('native',['mpiexec','-n',1,B/'evolve',12,'boost',2,1/24000,prefix,'-','uniform',cfg])
report=json.loads((O/'state.report.json').read_text());assert report['checkpoint_admitted'] and report['status']=='completed' and report['steps']==2 and report['dt']==1/24000
v=json.loads(subprocess.check_output([str(R/'output/refined-convergence/verify-checkpoint'),str(O/'state.checkpoint')],text=True));assert v['identity']==report['identity'] and v['leaves']==64 and v['n']==12
v.update(binary_sha256=sha(B/'evolve'),checkpoint_sha256=sha(O/'state.checkpoint'),checkpoint_bytes=(O/'state.checkpoint').stat().st_size,config_sha256=sha(cfg),original_smr_config_sha256=sha(config),qualification='Two-step local Serial interface/resolution control with identical uniform coarse core, physical data, angular refinement and timestep. Fourfold radial refinement .78-.90 only. Not full puncture-SMR qualification, GPU evidence, 10M stability or resolution convergence.')
(E/'verification.json').write_text(json.dumps(v,indent=2)+'\n')
for name in ['state.report.json','state.history.jsonl','uniform-core.cfg']:shutil.copy2(O/name,E/name)
shutil.copy2(B/'build.json',E/'build.json');print('verified',flush=True)
