#!/usr/bin/env python3
"""Actual autonomous scalar RK evolution: cadence and long aligned restart."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);p.add_argument('--output',type=Path);a=p.parse_args()
if a.output is None:a.output=Path(tempfile.mkdtemp(prefix='elliptic-clock-'))/'runs'
a.output.mkdir(parents=True,exist_ok=False)
common=[sys.executable,str(Path(__file__).with_name('evolve.py')),'--library',a.library,'--fast-library',a.fast_library,
 '--backend','spherical-g1','--n','64','--cfl','.05','--metric-variable','log','--lapse-variable','log-reference',
 '--elliptic-method','discrete-ode','--elliptic-bias','right','--endpoint','analytic-g1','--origin-ko','.5',
 '--lapse-restoring','1','--scri-sat','4','--regularity-sat','8','--scalar','--scalar-amplitude','.001','--amplitude','.001']
for name,target,cadence,parent in [('full-a',20.,.73,None),('full-b',20.,1.,None),('first',10.,.31,None),('resumed',20.,.73,'first')]:
 command=common+['--t',str(target),'--report-dt',str(cadence),'--output',str(a.output/name)]
 if parent:command+=['--resume',str(a.output/parent/'checkpoint.npz')]
 with (a.output/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,(name,(a.output/(name+'.log')).read_text()[-1500:])
 m=json.loads((a.output/name/'manifest.json').read_text());assert m['status']=='finished' and m['time']==target
 dt=.05/63.5;expected=round(target/dt);assert m['steps']==expected
 rows=[json.loads(s) for s in (a.output/name/'history.jsonl').read_text().splitlines()]
 assert all(abs(row['t']-row['step']*dt)<1e-13 for row in rows)
 if name=='full-b':np.testing.assert_allclose([row['t'] for row in rows],np.arange(21),atol=1e-13,rtol=0)
 for profile in (a.output/name).glob('profile-*.npz'):
  with np.load(profile,allow_pickle=False) as z:
   epoch=json.loads(str(z['time_lattice']));represented=epoch['origin_time']+(int(z['step'])-epoch['origin_step'])*epoch['dt']
   assert abs(float(z['t'])-represented)<1e-13
 with np.load(a.output/name/'checkpoint.npz',allow_pickle=False) as z:
  now={key:z[key].copy() for key in ['q','r','t','step','q_physical','chi_total','alpha_total']};meta=json.loads(str(z['metadata']))
 if name=='full-a':reference=now;energy=meta['radiated_energy']
 elif name in ['full-b','resumed']:
  for key in reference:assert np.array_equal(reference[key],now[key]),(name,key)
  assert energy==meta['radiated_energy']
record=dict(bitwise_state_time_grid_step_physical_totals_and_flux=True,steps=25400,time=20.,
 cadence_controls=[.73,1.],aligned_restart_time=10.,scope='Clock/reporting and restart consistency, not scalar stability/convergence admission')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
