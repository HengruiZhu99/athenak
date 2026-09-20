#!/usr/bin/env python3
import importlib.util,json
from pathlib import Path
import numpy as np
from mode_operator import *
op=Operator('validation')
res={}
z=op.advance(steps=2,label='zero_two_steps')
res['zero_two_steps']={'all_zero_bits':bool(np.all(z.view(np.uint64)==0)),'max_abs':float(abs(z).max())}
pulse=op.advance(steps=0,label='initial_pulse',overrides=('problem/vacuum_gauge_pulse_amplitude=1e-6',))
f=op.advance(pulse,steps=1,label='pulse_one_step')
f2=op.advance(f,steps=1,label='pulse_composed_two_steps')
f2direct=op.advance(pulse,steps=2,label='pulse_direct_two_steps')
res['composition_pulse']=discrepancy(f2direct,f2)
base=ROOT.parents[1]
helper=base/'evolution-debug-20260917/check-checkpoint.py'
spec=importlib.util.spec_from_file_location('check_checkpoint',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
checkpoint=base/'stability-isolation-20260919/gauge-slow/shift_Gamma2/rst/rank_00000000/ks_background.00002.rst'
meta=mod.inspect(checkpoint,scan=True)
seed=np.fromfile(checkpoint,dtype=np.float64,offset=checkpoint.stat().st_size-np.prod(SHAPE)*8).reshape(SHAPE)
seed.tofile(ROOT/'late_seed_500M.bin')
res['seed_checkpoint']=dict(path=str(checkpoint),metadata=meta,profile=profile(seed))
v=seed/max(abs(seed).max(),1e-99)*1e-6
f=op.advance(v,steps=2,label='seed_two_steps')
f2=op.advance(f,steps=2,label='seed_composed_four_steps')
f2direct=op.advance(v,steps=4,label='seed_direct_four_steps')
res['composition_late_seed']=discrepancy(f2direct,f2)
responses={}
for eps in [1e-5,3e-6,1e-6,3e-7]:
    r=op.response(seed,steps=40,epsilon=eps,label=f'linearity_{eps:g}')
    responses[eps]=r
    r.tofile(ROOT/f'response40_eps{eps:g}.bin')
res['amplitude_convergence']={str(e):discrepancy(responses[e],responses[1e-6]) for e in responses if e!=1e-6}
# Additivity includes an independent lapse pulse direction of the same norm.
a=seed/np.linalg.norm(seed);b=pulse/np.linalg.norm(pulse)
ra=responses[1e-6]/np.linalg.norm(seed)
rb=op.response(b,steps=40,epsilon=1e-6,label='additivity_b')
rab=op.response(a+b,steps=40,epsilon=1e-6,label='additivity_ab')
res['additivity']=discrepancy(ra+rb,rab)
(ROOT/'operator-validation.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k!='seed_checkpoint'},indent=2),flush=True)
