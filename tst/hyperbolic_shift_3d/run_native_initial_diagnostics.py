"""Collect already-completed native t=0 runs and measure common-cutoff norms.
Run from repository root; refuses to overwrite radial outputs. Does not evolve.
"""
from pathlib import Path
import subprocess,json,hashlib,shutil
out=Path('docs/hyperbolic-shift-3d/reproduction/native-shared-transition-v1/checks/initial-grid');out.mkdir(exist_ok=True)
for n in [16,24,32]:
 prefix=Path(f'output/native-shared-transition/initial-n{n}/puncture')
 report=Path(str(prefix)+'.report.json');cp=Path(str(prefix)+'.checkpoint')
 r=json.loads(report.read_text());assert r['status']=='completed' and r['t']==0 and r['steps']==0 and r['checkpoint_admitted'] and r['initial_data_provider']=='native_isotropic_wormhole_quadrature'
 for suffix in ['report.json','history.csv']:
  shutil.copyfile(Path(str(prefix)+'.'+suffix),out/f'n{n}.{suffix}')
 h=hashlib.sha256()
 with cp.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 args=['mpiexec','-n','1','build-native-shared-transition/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_checkpoint_constraints_native_control',str(n),str(cp),str(.0005*16/n),'.063',str(out/f'n{n}-radial.json'),'0,.03,.063,.13,.16,.18,.2,.3,.4,.45,.56,.6,.675,.7,.75,.775,.8,.9,.95,1']
 (out/f'n{n}-provenance.json').write_text(json.dumps(dict(checkpoint=str(cp),bytes=cp.stat().st_size,sha256=h.hexdigest(),command=args),indent=2)+'\n')
 with (out/f'n{n}-radial.log').open('w') as log:p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
 print(n,p.returncode,flush=True);assert p.returncode==0
