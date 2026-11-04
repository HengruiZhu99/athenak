from pathlib import Path
import subprocess,json,time,hashlib
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists());out=root/'output/refined-convergence/reference-radial18-two-step';out.mkdir(exist_ok=False)
cfg=root/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial18_reference.cfg';binary=root/'build-refined-convergence/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_refined_amr'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cmd=['mpiexec','-n','2',str(binary),'16','boost','2','.000125',str(out/'state'),'-','smr',str(cfg)]
r=dict(command=cmd,source='b52f09723b9c5a51340401bf91a7bf4426d8f339',config_sha256=sha(cfg),binary_sha256=sha(binary),started=time.time(),qualification='Local two-step CFL/readiness probe on new radial18 layout. Nonproduction native driver explicitly selects allowed initial dt under requested .000125 cap; selected dt will be recorded. No Aurora campaign change or long-time stability claim.')
(out/'command.json').write_text(json.dumps(r,indent=2)+'\n')
with (out/'native.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
r.update(returncode=p.returncode,elapsed=time.time()-r['started']);(out/'command.json').write_text(json.dumps(r,indent=2)+'\n');assert p.returncode==0
v=json.loads(subprocess.check_output([str(root/'output/refined-convergence/verify-checkpoint'),str(out/'state.checkpoint')],text=True));assert v['verified'] and v['steps']==2
report=json.loads((out/'state.report.json').read_text());assert report['checkpoint_admitted'] and v['identity']==report['identity'] and v['dt']==report['dt'];r.update(actual_dt=v['dt'],checkpoint_sha256=sha(out/'state.checkpoint'))
(out/'verification.json').write_text(json.dumps(v,indent=2)+'\n');(out/'command.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
