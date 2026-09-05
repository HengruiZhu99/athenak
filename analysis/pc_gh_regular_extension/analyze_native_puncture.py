"""Uncensored native-cell puncture profiles, without Cartesian interpolation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vis/python'))
import bin_convert


def cells(path):
    data = bin_convert.read_binary(str(path))
    positions, spacings, states = [], [], []
    for m in range(data['n_mbs']):
        bounds = data['mb_geometry'][m].reshape(3, 2)
        counts = np.array([data[f'nx{d}_mb'] for d in [1, 2, 3]])
        dx = (bounds[:, 1]-bounds[:, 0])/counts
        indices = data['mb_index'][m].reshape(3, 2)
        axes = [bounds[d, 0]+(np.arange(lo, hi+1)+.5)*dx[d]
                for d, (lo, hi) in enumerate(indices)]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
        xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        positions.append(xyz)
        spacings.append(np.tile(dx, (len(xyz), 1)))
        states.append(np.column_stack([data['mb_data'][name][m].ravel()
                                       for name in data['var_names']]))
    return data, np.vstack(positions), np.vstack(spacings), np.vstack(states).astype(float)


def symmetry(xyz, u, names):
    """Check tensor reflections and x/y exchange on the native cell set.

    Binary field output has float32 precision. These are slice checks; z
    reflection and arbitrary rotations require additional native samples.
    """
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    slots = [('w', ())] + [('g', ab) for ab in pairs] + [('K', ())]
    slots += [('A', ab) for ab in pairs] + [('Z', (a,)) for a in range(3)]
    slots += [('C', ()), ('rho', ())]
    slots += [(family, (a,)) for family in ['beta', 'p'] for a in range(3)]
    slots += [('Q', (i,)+ab) for i in range(3) for ab in pairs]
    slots += [('L', (i,)) for i in range(3)]
    slots += [('B', (i, a)) for i in range(3) for a in range(3)]
    lookup = {tuple(np.round(point, 12)): i for i, point in enumerate(xyz)}
    if len(lookup) != len(xyz):
        raise ValueError('Duplicate native cell positions')
    results = {}
    for operation in ['reflect_x', 'reflect_y', 'swap_xy']:
        target = xyz.copy()
        if operation == 'swap_xy':
            target = target[:, [1, 0, 2]]
            permutation = []
            for family, indices in slots:
                changed = tuple({0: 1, 1: 0, 2: 2}[i] for i in indices)
                if family in ['g', 'A']:
                    changed = tuple(sorted(changed))
                elif family == 'Q':
                    changed = (changed[0],)+tuple(sorted(changed[1:]))
                permutation.append(slots.index((family, changed)))
            expected = u[:, permutation]
        else:
            axis = 0 if operation == 'reflect_x' else 1
            target[:, axis] *= -1
            expected = u*np.array([(-1)**indices.count(axis) for _, indices in slots])
        try:
            partner = [lookup[tuple(np.round(point, 12))] for point in target]
        except KeyError as error:
            raise ValueError(f'Native slice does not support {operation}') from error
        error = np.abs(u[partner]-expected).max(axis=0)
        results[operation] = dict(max_abs=float(error.max()),
                                  by_component=dict(zip(names, map(float, error))))
    return results


def analyze(path, output):
    data, xyz, dx, u = cells(path)
    if not np.isfinite(u).all():
        raise ValueError(f'Nonfinite native data: {path}')
    if data['var_names'][0] != 'pcgh_w' or u.shape[1] != 55:
        raise ValueError('This analyzer requires all 55 regular PC-GH variables')
    r = np.linalg.norm(xyz, axis=1)
    if np.any(r == 0):
        raise ValueError('A native sample lies exactly on r=0; inspect separately')
    h = float(dx.min())
    fields = dict(w=u[:, 0], rho=u[:, 18], alpha=u[:, 0]*u[:, 18], K=u[:, 7],
                  Cperp=u[:, 17])
    for name, lo, hi in [('Atilde', 8, 14), ('Z', 14, 17), ('beta', 19, 22),
                         ('p', 22, 25), ('Q', 25, 43), ('L', 43, 46), ('B', 46, 55)]:
        fields[name] = np.linalg.norm(u[:, lo:hi], axis=1)
    g = np.empty((len(u), 3, 3))
    for index, (a, b) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        g[:, a, b] = g[:, b, a] = u[:, 1+index]
    eig = np.linalg.eigvalsh(g)
    result = dict(time=float(data['time']), cycle=int(data['cycle']), min_spacing=h,
                  min_sample_radius=float(r.min()), cell_count=len(r),
                  min_metric_eigenvalue=float(eig.min()), min_w=float(u[:, 0].min()),
                  min_rho=float(u[:, 18].min()), max_alpha=float(fields['alpha'].max()),
                  max_alpha2_chi=float((fields['alpha']**2*u[:, 0]**2).max()),
                  scope='native cells in the supplied slice, not a full-volume maximum',
                  field_output_precision='float32, promoted for analysis',
                  symmetry=symmetry(xyz, u, data['var_names']), fields={})
    edges = np.geomspace(.999*r.min(), 1.0, 17)
    records = []
    for name, values in fields.items():
        magnitude = np.abs(values)
        points = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (r >= lo) & (r < hi)
            if mask.any():
                # Store actual sampled radius of the maximum, not an invented
                # midpoint in an empty or poorly resolved radial shell.
                indexes = np.flatnonzero(mask)
                winner = indexes[np.argmax(magnitude[mask])]
                points.append((r[winner], magnitude[winner]))
                records.append((name, lo, hi, r[winner], magnitude[winner], len(indexes)))
        usable = [(radius, value) for radius, value in points
                  if 2*h <= radius <= .5 and value > 0]
        slope = float(np.polyfit(np.log([p[0] for p in usable]),
                                 np.log([p[1] for p in usable]), 1)[0]) if len(usable) >= 4 else None
        near = r <= 2*h
        inner = r <= .5
        result['fields'][name] = dict(max_abs=float(magnitude.max()),
            nearest_2h_max=float(magnitude[near].max()) if near.any() else None,
            fixed_r05_max=float(magnitude[inner].max()) if inner.any() else None,
            fitted_inner_power=slope, power_fit_bins=len(usable))
    output.mkdir(parents=True, exist_ok=True)
    (output/(path.stem+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    with (output/(path.stem+'-profiles.csv')).open('w') as file:
        file.write('field,r_left,r_right,sampled_radius,max_abs,count\n')
        for row in records:
            file.write(','.join(map(str, row))+'\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('files', type=Path, nargs='+')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    for path in args.files:
        print(path, json.dumps(analyze(path, args.output), sort_keys=True))
