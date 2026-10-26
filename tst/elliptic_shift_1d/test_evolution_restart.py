#!/usr/bin/env python3
"""Executable scalar/noise checkpoint consistency using aligned RK timesteps."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library');p.add_argument('--output',type=Path,required=True)
p.add_argument('--scenario',choices=['scalar-noise','puncture-chart','trumpet-log'],default='scalar-noise')
p.add_argument('--slicing-family',choices=['linear','rational'],default='linear')
p.add_argument('--radial-operator',choices=['fd','sbp63'],default='fd')
p.add_argument('--lapse-reference',choices=['minkowski','CMC-trumpet'],default='minkowski')
p.add_argument('--metric-variable',choices=['deviation','log-trumpet'],default='deviation')
p.add_argument('--elliptic-quadrature',choices=['prefix','inward'],default='prefix')
p.add_argument('--elliptic-method',choices=['quadrature','discrete-ode'],default='quadrature')
p.add_argument('--lapse-restoring-profile',choices=['constant','physical-log'],default='constant')
p.add_argument('--lapse-background-source',choices=['reference','collapsed-inner'],default='reference')
p.add_argument('--lapse-restoring-form',choices=['symmetric','multiplicative'],default='symmetric')
p.add_argument('--positive-variables',action='store_true')
p.add_argument('--subtract-local',action='store_true')
p.add_argument('--compensated-update',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
dt=.05/63.5;driver=Path(__file__).with_name('evolve.py')
common=[sys.executable,str(driver),'--library',a.library,'--backend','spherical-g1',
    '--n','64','--scalar','--lapse-restoring','1','--scri-sat','4',
    '--regularity-sat','8','--origin-ko','.5','--perturbation','noise',
    '--perturbation-amplitude','1e-12']
if a.positive_variables:
    if a.scenario!='scalar-noise':raise ValueError('Regular positive-variable restart uses scalar/noise')
    common+=['--metric-variable','log','--lapse-variable','log-reference']
if a.scenario=='puncture-chart':
    dt=.0625/64
    common=[sys.executable,str(driver),'--library',a.library,'--backend','spherical-g1',
        '--n','65','--cfl','.0625','--puncture','--puncture-chart','--initial-data','wormhole',
        '--lapse-restoring','1','--lapse-restoring-form','multiplicative','--scri-sat','4',
        '--regularity-sat','8','--puncture-ko','.5','--slicing-family',a.slicing_family,'--radial-operator',a.radial_operator]
if a.scenario=='trumpet-log':
    if a.fast_library:raise ValueError('The log-trumpet lapse prototype uses Python RK')
    dt=.05/64.5
    common=[sys.executable,str(driver),'--library',a.library,'--backend','spherical-g1',
        '--n','65','--puncture','--initial-data','trumpet','--mass',str(1/3),'--lapse-variable','log-trumpet',
        '--lapse-reference',a.lapse_reference,'--metric-variable',a.metric_variable,'--lapse-restoring','1','--lapse-restoring-form','multiplicative',
        '--scri-sat','4','--regularity-sat','8','--amplitude','.001']
if a.fast_library:common+=['--fast-library',a.fast_library]
common+=['--elliptic-quadrature',a.elliptic_quadrature,'--elliptic-method',a.elliptic_method]
common+=['--lapse-restoring-profile',a.lapse_restoring_profile,'--lapse-background-source',a.lapse_background_source]
if a.scenario=='scalar-noise':common+=['--lapse-restoring-form',a.lapse_restoring_form]
if a.subtract_local:common+=['--subtract-local']
if a.compensated_update:common+=['--compensated-update']
for name,steps,resume in [('full',40,None),('first',20,None),('resumed',40,'first')]:
    command=common+['--t',str(steps*dt),'--output',str(a.output/name)]
    if resume:command+=['--resume',str(a.output/resume/'checkpoint.npz')]
    result=subprocess.run(command,text=True,capture_output=True)
    (a.output/(name+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,(name,result.stderr)
    manifest=json.loads((a.output/name/'manifest.json').read_text())
    assert manifest['status']=='finished',manifest
with np.load(a.output/'full/checkpoint.npz',allow_pickle=False) as x,np.load(a.output/'resumed/checkpoint.npz',allow_pickle=False) as y:
    keys=['q','r','t','step']
    if a.compensated_update:keys+=['update_compensation']
    if a.scenario in ('puncture-chart','trumpet-log') or a.positive_variables:keys+=['q_physical','chi_total','alpha_total']
    for key in keys:assert np.array_equal(x[key],y[key]),key
    mx=json.loads(str(x['metadata']));my=json.loads(str(y['metadata']))
    assert mx['radiated_energy']==my['radiated_energy']
record=dict(bitwise_state_time_grid_step_and_flux=True,steps=40,n=64 if a.scenario=='scalar-noise' else 65,
    scenario=a.scenario,slicing_family=a.slicing_family,compared_keys=keys,
    scope='Short restart consistency; no stability assertion')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
