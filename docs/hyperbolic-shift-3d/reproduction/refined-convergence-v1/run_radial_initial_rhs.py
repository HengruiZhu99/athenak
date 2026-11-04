"""Reproduce preserved local controls; refuses existing output records."""
from pathlib import Path
import subprocess,json,time,hashlib
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists());out=root/'output/refined-convergence/reference-radial18-initial';out.mkdir(exist_ok=False)
base=root/'output/refined-convergence/initial-native-rhs-probe-dt';cfg=root/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial18_reference.cfg';binary=root/'build-refined-convergence/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_refined_amr'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
commands=[['mpiexec','-n','2',str(binary),'16','boost','0','.00003125',str(out/'state'),'-','smr',str(cfg)]]
record=dict(commands=commands,config_sha256=sha(cfg),production_binary_sha256=sha(binary),analysis_binary_sha256=sha(base/'reader'),started=time.time(),qualification='Initial-only local radial-refinement control, not Aurora evolution; only .90-.95 radial spacing changes, angular/core/physics/scri final shell unchanged. Smaller dt is a zero-step characteristic probe.')
(out/'commands.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'initial.log').open('w') as f:p=subprocess.run(commands[0],stdout=f,stderr=subprocess.STDOUT,timeout=600)
assert p.returncode==0
v=json.loads(subprocess.check_output([str(root/'output/refined-convergence/verify-checkpoint'),str(out/'state.checkpoint')],text=True));assert v['verified'] and v['steps']==0 and v['time']==0
(out/'checkpoint-verification.json').write_text(json.dumps(v,indent=2)+'\n');record['checkpoint_sha256']=sha(out/'state.checkpoint');record['initial_finished']=time.time();print('initial radial18 complete',flush=True)
# RHS analysis uses a much smaller probe only for this finer radial grid.
# Reader takes dt from the exact admitted checkpoint, preserving its CFL.
command=['mpiexec','-n','2',str(base/'reader'),'16',str(out/'state.checkpoint'),str(cfg),str(out/'rhs'),'reconstruct-initial']
record['commands'].append(command);(out/'commands.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'rhs.log').open('w') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=600)
record.update(rhs_returncode=p.returncode,finished=time.time(),checkpoint_unchanged=sha(out/'state.checkpoint')==record['checkpoint_sha256']);(out/'commands.json').write_text(json.dumps(record,indent=2)+'\n');assert p.returncode==0 and record['checkpoint_unchanged']
print('radial18 RHS complete',flush=True)
