"""Three-resolution exterior state differences on the common Cartesian slice."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from make_inputs import parse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis/pc_gh_localization'))
from plot_qualification import nearest_cart_files


def richardson_order(h, ratio):
    a, b = np.log(h[0]/h[1]), np.log(h[1]/h[2])
    def model(p):
        return a/b if abs(p) < 1e-8 else np.expm1(p*a)/(-np.expm1(-p*b))
    left, right = -20., 20.
    if not model(left) <= ratio <= model(right):
        return None
    for _ in range(80):
        mid = (left+right)/2
        if model(mid) < ratio:
            left = mid
        else:
            right = mid
    return (left+right)/2


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('runs', nargs=3, type=Path)
ap.add_argument('--time', type=float, default=6)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
states = []
for run in args.runs:
    params = parse((run/'used_input.athinput').read_text())
    h = float(params['problem']['expected_finest_spacing'])
    found = nearest_cart_files(run/'cart', 'pcgh_slice', (args.time,))
    if not found or abs(found[0][0]-args.time) > 1e-4:
        raise ValueError(f'Missing common-time Cartesian state in {run}')
    state = found[0][2]
    u = np.stack(list(state['data'].values()))[:, 0].astype(float)
    if u.shape[0] != 55 or not np.isfinite(u).all():
        raise ValueError(f'Nonfinite or incomplete regular state in {run}')
    states.append((h, state, u, run))
states.sort(key=lambda item: -item[0])
h = np.array([s[0] for s in states])
if not (h[0] > h[1] > h[2]):
    raise ValueError('Three distinct spacings required')
for _, state, _, _ in states[1:]:
    if any(not np.array_equal(state[d], states[0][1][d]) for d in ['x', 'y', 'z']):
        raise ValueError('Cartesian sample grids differ')
x, y = np.meshgrid(states[0][1]['x'], states[0][1]['y'])
radius = np.hypot(x, y)
groups = dict(w=(0,1), gtilde=(1,7), K=(7,8), Atilde=(8,14), Z=(14,17),
              Cperp=(17,18), rho=(18,19), beta=(19,22), p=(22,25), Q=(25,43),
              L=(43,46), B=(46,55), all=(0,55))
result = dict(time=args.time, runs=[str(s[3]) for s in states], finest_spacing=h.tolist(),
              scope='Interpolated common Cartesian exterior slice, not a native puncture-power test',
              regions={})
for inner, outer in [(1,1.75), (2,2.5)]:
    selected = (radius >= inner) & (radius <= outer)
    region = {}
    for name, (lo, hi) in groups.items():
        differences = [(states[i][2][lo:hi]-states[i+1][2][lo:hi])[:, selected].ravel()
                       for i in [0,1]]
        norms = np.array([np.linalg.norm(d) for d in differences])
        rms = norms/np.sqrt(selected.sum())
        ratio = float(norms[0]/norms[1]) if norms[1] else None
        region[name] = dict(rms_differences=rms.tolist(), ratio=ratio,
            fitted_order=richardson_order(h, ratio) if ratio else None,
            difference_alignment=float(np.dot(*differences)/np.prod(norms)) if norms.all() else None)
    result['regions'][f'{inner}<=r<={outer}'] = region
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2)+'\n')
for region, groups in result['regions'].items():
    print(region, {name: round(v['fitted_order'],3) if v['fitted_order'] is not None else None
                   for name,v in groups.items()})
