#!/usr/bin/env python3
"""Same-binary physical-field time/solve/arithmetic isolation at saved times."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analyze_principal_controls import load,spatial,norm,constraint_convergence

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
zs=[load(a.input/f'spatial-n{n}') for n in [2048,4096,8192]];base=zs[1];space=spatial(zs)
def frames(folder):
    result={}
    for path in sorted(folder.glob('profile-*.npz')):
        with np.load(path,allow_pickle=False) as z:
            t=float(z['t']);tick=int(round(t*10))
            if abs(t-tick/10)>1e-10:raise ValueError('Unexpected profile time')
            result[tick]=(z['q_physical'].copy(),hashlib.sha256(path.read_bytes()).hexdigest())
    return result
baseline=frames(a.input/'spatial-n4096');controls={};run_receipts=[]
for name,changed in [('half-dt-n4096','cfl'),('solve0-n4096','elliptic_refinements'),
                     ('solve4-n4096','elliptic_refinements'),('uncompensated-n4096','compensated_update')]:
    z=load(a.input/name)
    expected=base['manifest']['binding'];actual=z['manifest']['binding']
    inactive=[]
    if name=='uncompensated-n4096':
        # This Python helper is source-bound only when Kahan is enabled.
        # It is unused by the uncompensated run; all compiled bytes and all
        # active source files must still match exactly.
        helper='tst/elliptic_shift_1d/compensated_update.py'
        expected=json.loads(json.dumps(expected));digest=expected['source_sha256'].pop(helper)
        frozen=a.input/'matrix-source'/helper
        if hashlib.sha256(frozen.read_bytes()).hexdigest()!=digest:raise ValueError('Inactive helper differs from the frozen baseline')
        inactive=[dict(path=helper,sha256=digest,reason='Inactive in uncompensated run')]
    if actual!=expected:raise ValueError('Different active numerical source/binary: '+name)
    if z['t']!=base['t'] or not np.array_equal(z['r'],base['r']):raise ValueError('Different comparison time/grid')
    for key,value in base['manifest']['arguments'].items():
        if key not in [changed,'output'] and z['manifest']['arguments'][key]!=value:raise ValueError('Uncontrolled input change: '+key)
    saved=frames(a.input/name)
    if set(saved)!=set(baseline) or set(saved)!=set(range(11)):raise ValueError('Incomplete common trajectory coverage')
    result={}
    for region in ['bulk','endpoint']:
        mask=base['r']<.8 if region=='bulk' else base['r']>.9
        differences=[norm(saved[t][0]-baseline[t][0],base['r'],mask) for t in sorted(saved)]
        fine_difference=space['regions'][region]['differences'][1]
        order=space['regions'][region]['order'];ratio=space['h'][1]/space['h'][2]
        # Middle-grid truncation estimate, since controls run at the middle grid.
        estimate=fine_difference/(1-ratio**(-order)) if order is not None and order>0 else None
        result[region]=dict(sampled_differences=differences,maximum_difference=max(differences),final_difference=differences[-1],
            raw_middle_to_finest_spatial_difference=fine_difference,Richardson_middle_spatial_error=estimate,
            final_error_fraction=None if estimate is None else differences[-1]/estimate)
    controls[name]=dict(changed_input=changed,old=base['manifest']['arguments'][changed],new=z['manifest']['arguments'][changed],regions=result,
        stages=z['stages'],checkpoint_sha256=z['checkpoint_sha256'],segments=z['segments'],verified_inactive_source=inactive,
        profile_hashes={str(t):saved[t][1] for t in saved})
record=dict(spatial=space,constraints=constraint_convergence(zs),controls=controls,
    runs=[{key:z[key] for key in ['folder','checkpoint_sha256','segments','stages']} for z in zs],binding=base['manifest']['binding'],
    baseline_profile_hashes={str(t):baseline[t][1] for t in baseline},
    source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in [Path(__file__).name,'analyze_principal_controls.py','trajectory.py']},
    scope='Finite A=.1 pulse through1, full exact saved-time comparison and every-step/stage extrema. One timestep halving isolates a difference but cannot establish temporal order. Richardson estimates require the observed spatial regime and are reported alongside raw differences. Finest constraint floors and all failed gates remain visible; no long-duration or principal admission.')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({name:value['regions'] for name,value in controls.items()}))
