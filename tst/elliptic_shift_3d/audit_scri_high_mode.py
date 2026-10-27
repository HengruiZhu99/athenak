#!/usr/bin/env python3
"""Measure a highest radial polynomial contribution; never modify a solved field."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg
from compatible_sbp import mm
from mapped_sbp_probe import ShellGrid
from affine_sbp_probe import AffineConsistentGrid
from compatible_tensor_gauge import TensorGauge

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
root=Path(__file__).resolve().parents[2]
case=root/'docs/elliptic-shift-3d/evidence/compatible-wormhole-affine-buffer-n10-20261027/result'
assessment=json.loads((case/'assessment.json').read_text());profile=np.load(case/'profile.npz')
grid=AffineConsistentGrid(ShellGrid(assessment['n'],radial=tuple(assessment['radial_partition_s']),transfer_seed='zero'))
if np.max(abs(grid.coordinates-profile['coordinates']))!=0:raise ValueError('diagnostic grid differs from saved solution')
gauge=TensorGauge(grid,np.broadcast_to(np.eye(3),grid.weights.shape+(3,3)).copy())
h=gauge.rate(profile['beta']);highest=np.zeros_like(h);n=grid.n
for b,axis,side in grid.outer:
    if axis!=0 or side!=1:raise ValueError('unexpected outer radial topology')
    V=leg.legvander(2*grid.locals[b]['x']-1,n-1)
    coefficients=mm(np.linalg.inv(V),h[b].reshape(n,-1))
    highest[b]=(V[:,-1,None]*coefficients[-1,None,:]).reshape(h[b].shape)
contribution=gauge.variation(highest);mask=profile['boundary']
original=profile['actual_metric_rate'][mask];part=contribution[mask];remainder=original-part
norm=lambda value:float(np.sqrt(np.sum(value*value)))
result={'scope':__doc__,'case_sha256':hashlib.sha256((case/'assessment.json').read_bytes()).hexdigest(),'profile_sha256':hashlib.sha256((case/'profile.npz').read_bytes()).hexdigest(),
        'highest_radial_degree':n-1,'original_scri_max':float(np.max(abs(original))),
        'highest_contribution_scri_max':float(np.max(abs(part))),
        'remainder_scri_max':float(np.max(abs(remainder))),
        'highest_contribution_relative_l2':norm(part)/norm(original),
        'remainder_relative_l2':norm(remainder)/norm(original),
        'signed_contribution_fraction':float(np.sum(part*original)/np.sum(original*original)),
        'interpretation':'Unweighted vector L2 over every scri node. This is a linear diagnostic decomposition of a saved failure, not a filtered solution or a correction.'}
(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
