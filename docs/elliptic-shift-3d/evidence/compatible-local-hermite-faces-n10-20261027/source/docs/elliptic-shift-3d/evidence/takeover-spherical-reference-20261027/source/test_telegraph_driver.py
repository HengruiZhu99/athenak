#!/usr/bin/env python3
"""End-to-end initial invariant mass and exact checkpoint continuation."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args()
base=[sys.executable,str(Path(__file__).with_name('run_telegraph.py')),'--lapse-floor','0','--n','33','--ray-start','0','--report-dt','.015625','--library',a.library,'--fast-library',a.fast_library]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 def run(name,*args):
  out=root/name;subprocess.run([*base,'--output',str(out),*args],check=True,capture_output=True,text=True);return out
 for mass in [1/3,.2]:
  out=run('mass'+str(mass),'--mass',str(mass),'--t','0')
  with np.load(out/'initial-data.npz') as z:
   r=z['r'];mask=(r>.1)&(r<.8);r=r[mask];q=z['q_physical'][mask];d=z['d'][mask];O=(1-r*r)/2;chi=1+q[:,0]
   R=r/(O*np.sqrt(chi));K=q[:,3]-3+2*q[:,4]
   normal=R*(K/3-O*q[:,2]/2);spatial=(1+r*r)/(2*O)-r*d[:,0]/(2*chi)
   m=R/2*(1+normal*normal-spatial*spatial)
   np.testing.assert_allclose(m,mass,atol=2e-12,rtol=2e-12)
 for case,profile in [('collar','collar'),('sinh','af-regular'),('local','af-regular'),('high-order','af-regular'),('scri-map','af-regular'),('pchip','af-regular'),('pchip-dense','af-regular')]:
  controls=['--inner-profile',profile]
  if case=='sinh':controls+=['--quadrature-width','9','--puncture-ko','.25','--grid-stretch','4']
  if case=='local':controls+=['--grid-stretch','14','--grid-map','inner-c7','--mapped-ko','local']
  if case=='high-order':controls+=['--n','129','--width','9','--ko-order','8','--grid-stretch','16','--grid-map','inner-exp','--mapped-ko','local']
  if case=='pchip-dense':controls+=['--ray-interpolation','pchip','--ray-exit','dense','--grid-stretch','4']
  if case=='pchip':controls+=['--ray-interpolation','pchip','--grid-stretch','4']
  if case=='scri-map':controls+=['--grid-stretch','0','--grid-map','inner-exp-scri','--mapped-ko','local','--ko-order','8']
  whole=run(case+'-whole',*controls,'--t','.03125');first=run(case+'-first',*controls,'--t','.015625')
  split=run(case+'-split',*controls,'--t','.03125','--resume',str(first/'checkpoint.npz'))
  with np.load(whole/'checkpoint.npz') as x,np.load(split/'checkpoint.npz') as y:
   for key in ['q','B','rays','compensation','t','step']:assert np.array_equal(x[key],y[key]),key
 # The user default and nondefault floors are recorded and restart-bound.
 default=root/'floor-default'
 subprocess.run([base[0],base[1],*base[4:],'--output',str(default),'--t','0'],check=True,capture_output=True,text=True)
 assert json.loads((default/'manifest.json').read_text())['arguments']['lapse_floor']==1e-10
 old_controls=controls
 controls=['--n','129','--grid-stretch','4','--lapse-floor','.001','--report-dt','.000390625','--lapse-update','total','--puncture-stencil','positive','--lapse-attenuation','.0001']
 whole=run('floor-whole',*controls,'--t','.00078125');part=run('floor-part',*controls,'--t','.000390625');split=run('floor-split',*controls,'--t','.00078125','--resume',str(part/'checkpoint.npz'))
 with np.load(whole/'checkpoint.npz') as x,np.load(split/'checkpoint.npz') as y:
  for key in ['q','B','rays','compensation','t','step']:np.testing.assert_array_equal(x[key],y[key])
 bad=subprocess.run([*base,'--output',str(root/'bad-floor'),*controls,'--lapse-floor','.002','--t','.00078125','--resume',str(part/'checkpoint.npz')],capture_output=True,text=True)
 assert bad.returncode!=0 and 'Restart input mismatch: lapse_floor' in bad.stderr
 bad_mode=subprocess.run([*base,'--output',str(root/'bad-lapse-update'),*controls,'--lapse-update','log','--t','.00078125','--resume',str(part/'checkpoint.npz')],capture_output=True,text=True)
 assert bad_mode.returncode!=0 and 'Restart input mismatch: lapse_update' in bad_mode.stderr
 controls=old_controls
 # Exit modes are part of the restart binding.
 bad_mode=subprocess.run([*base,'--output',str(root/'bad-exit-mode'),*controls,'--ray-exit','legacy','--t','.03125','--resume',str(first/'checkpoint.npz')],capture_output=True,text=True)
 assert bad_mode.returncode!=0 and 'Restart input mismatch: ray_exit' in bad_mode.stderr
 # An unresolved N33 strong-map, wide-stencil evolution must stop on the
 # coupled-speed guard, rather than silently taking unsafe stages.
 unsafe=subprocess.run([*base,'--output',str(root/'underresolved'),'--inner-profile','af-regular','--width','9','--ko-order','8','--grid-stretch','16','--grid-map','inner-exp','--mapped-ko','local','--t','.03125'],capture_output=True,text=True)
 assert unsafe.returncode!=0
 failure=json.loads((root/'underresolved'/'manifest.json').read_text())['failure']
 assert 'timestep guard' in failure['message']
 # A bad later parent must be rejected before dispatching the valid first one.
 rejected=root/'bad-batch'
 result=subprocess.run([sys.executable,str(Path(__file__).with_name('continue_telegraph_matrix.py')),'--parents',str(first),str(root/'missing-parent'),'--t','.03125','--output',str(rejected)],capture_output=True,text=True)
 assert result.returncode!=0 and not rejected.exists(),'Continuation preflight dispatched a partial invalid batch'
 print(json.dumps(dict(initial_invariant_masses=[1/3,.2],cli_checkpoint_bitwise=True,inner_profiles=['collar','af-regular'],nondefault_quadrature_and_KO=True,auxiliary_and_stage_rays_included=True)))
