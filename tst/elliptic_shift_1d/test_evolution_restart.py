#!/usr/bin/env python3
"""Executable scalar/noise checkpoint consistency using aligned RK timesteps."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library');p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
dt=.05/63.5;driver=Path(__file__).with_name('evolve.py')
common=[sys.executable,str(driver),'--library',a.library,'--backend','spherical-g1',
    '--n','64','--scalar','--lapse-restoring','1','--scri-sat','4',
    '--regularity-sat','8','--origin-ko','.5','--perturbation','noise',
    '--perturbation-amplitude','1e-12']
if a.fast_library:common+=['--fast-library',a.fast_library]
for name,steps,resume in [('full',40,None),('first',20,None),('resumed',40,'first')]:
    command=common+['--t',str(steps*dt),'--output',str(a.output/name)]
    if resume:command+=['--resume',str(a.output/resume/'checkpoint.npz')]
    result=subprocess.run(command,text=True,capture_output=True)
    (a.output/(name+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,(name,result.stderr)
    manifest=json.loads((a.output/name/'manifest.json').read_text())
    assert manifest['status']=='finished',manifest
with np.load(a.output/'full/checkpoint.npz',allow_pickle=False) as x,np.load(a.output/'resumed/checkpoint.npz',allow_pickle=False) as y:
    for key in ['q','r','t','step']:assert np.array_equal(x[key],y[key]),key
    mx=json.loads(str(x['metadata']));my=json.loads(str(y['metadata']))
    assert mx['radiated_energy']==my['radiated_energy']
record=dict(bitwise_state_time_grid_step_and_flux=True,steps=40,n=64,scalar=True,
    supported_noise=True,scope='Short restart consistency; no stability assertion')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
