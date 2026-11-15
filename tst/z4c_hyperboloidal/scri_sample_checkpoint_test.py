#!/usr/bin/env python3
"""Verify common-point checkpoint sampling and measure its known-pulse error."""
import argparse,json
from pathlib import Path
import numpy as np
from scri_sample_checkpoint import weights,sample,coordinates,RADII,ANGLES
p=argparse.ArgumentParser();p.add_argument('--seeds',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);args=p.parse_args()
results={};worst=0
for n in (12,16,20):
    for width in (6,8):
        for nodes,targets,k in [((np.arange(n)+.5)/n,ANGLES,width),
                               (.94+np.arange(n+1)*(.06/n),RADII,width+1)]:
            w=weights(nodes,targets,k)
            for degree in range(k):
                worst=max(worst,float(np.max(abs(w@nodes**degree-targets**degree))))
    for mode in ('A','B'):
        zero=next((args.seeds/f'{mode}-N{n}-control/rst').glob('*.rst'))
        pulse=next((args.seeds/f'{mode}-N{n}-pulse/rst').glob('*.rst'))
        _,_,z=sample(zero,zero)
        assert all(np.count_nonzero(v)==0 for v in z.values())
        _,_,values=sample(pulse,zero)
        exact=np.zeros(len(RADII));x=(RADII-.92)/.04;inside=abs(x)<1
        exact[inside]=1e-6*np.exp(1-1/(1-x[inside]**2))
        results[f'{mode}-N{n}']={key:float(np.max(abs(v[:,:,:,:,18]-exact[None,:,None,None])))
                                for key,v in values.items()}
        for v in values.values():
            assert np.count_nonzero(v[:,:,:,:,[f for f in range(31) if f not in (13,18)]])==0
assert worst<1e-12
assert np.max(abs(np.linalg.norm(coordinates(),axis=-1)-RADII[None,:,None,None]))<1e-14
result=dict(polynomial_exactness_max_error=worst,zero_residual_exact=True,
            only_lapse_trace_changed=True,known_initial_pulse_interpolation_error=results,
            interpretation='Compare width sensitivity with measured solution differences; interpolation is not assumed negligible.')
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
