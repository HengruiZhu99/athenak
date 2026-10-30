#!/usr/bin/env python3
"""Compare separate Cartesian/shell fixed-ray samples to a same-time 1D profile.

The angular statistics are unweighted over the available rays at each radius;
they are not sphere integrals. No averaging across the two domain copies.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = ('chi', 'alpha', 'beta_r', 'g_rr', 'A_rr', 'Theta')


def radial_fields(data):
    normal = np.column_stack([data[k] / data['r'] for k in ('x', 'y', 'z')])
    assert np.allclose(np.sum(normal**2, axis=1), 1, rtol=0, atol=1e-12)
    values = {'chi': data['v0'], 'alpha': data['v18'] + (1 + data['r']**2) / 2,
              'beta_r': sum(normal[:, i]*data[f'v{22+i}'] for i in range(3)) - data['r'],
              'Theta': data['v14']}
    for name, start, background in [('g_rr', 1, 1.), ('A_rr', 7, 0.)]:
        values[name] = np.full(len(data), background)
        for f, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
            values[name] += (1 if i == j else 2)*normal[:, i]*normal[:, j]*data[f'v{start+f}']
    return values


def interpolate_six(x, y, locations):
    assert len(x) >= 6 and np.all(np.diff(x) > 0)
    assert np.min(locations) >= x[0] and np.max(locations) <= x[-1]
    result = []
    for point in locations:
        start = int(np.clip(np.searchsorted(x, point)-3, 0, len(x)-6))
        nodes = x[start:start+6]
        weights = np.ones(6)
        for i in range(6):
            for j in range(6):
                if i != j:
                    weights[i] *= (point-nodes[j])/(nodes[i]-nodes[j])
        result.append(np.dot(weights, y[start:start+6]))
    return np.array(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('profile', 'evolution-report', 'reference', 'reference-report', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    data = np.atleast_1d(np.genfromtxt(a.profile, delimiter=',', names=True))
    reference = np.atleast_1d(np.genfromtxt(a.reference, delimiter=',', names=True))
    report = json.loads(a.reference_report.read_text())
    evolution = json.loads(a.evolution_report.read_text())
    assert report['status'] == 'completed' and report['kind'] == 'puncture'
    assert evolution['status'] == 'completed' and evolution['kind'] == 'puncture'
    assert evolution['checkpoint_admitted'] and abs(evolution['t']-report['t']) < 1e-13
    assert np.isfinite(data.view(float)).all()
    # The donor's unused expansion diagnostic is undefined at scri. Require
    # finite coordinates and every field actually used in this comparison.
    assert all(np.isfinite(reference[key]).all() for key in ('r', *FIELDS))
    assert np.all(abs(data['t']-report['t']) < 1e-13), 'Comparison requires matching coordinate times'
    assert len(set(zip(data['domain'], data['ray'], data['r']))) == len(data)
    assert set(data['domain']) == {0, 1}
    values = radial_fields(data)
    matched = {key: interpolate_six(reference['r'], reference[key], data['r']) for key in FIELDS}
    a.out.mkdir(parents=True, exist_ok=False)
    rows, aggregates, summary = [], [], {}
    for i in range(len(data)):
        row = {key: float(data[key][i]) for key in ('t', 'domain', 'ray', 'r')}
        for key in FIELDS:
            row[key+'_3d'] = float(values[key][i])
            row[key+'_1d'] = float(matched[key][i])
            row[key+'_difference'] = float(values[key][i]-matched[key][i])
        rows.append(row)
    for domain in (0, 1):
        mask = data['domain'] == domain
        summary[str(domain)] = {key: float(np.max(abs(values[key][mask]-matched[key][mask]))) for key in FIELDS}
        for radius in np.unique(data['r'][mask]):
            keep = mask & (data['r'] == radius)
            row = dict(domain=domain, r=float(radius), ray_count=int(keep.sum()))
            for key in FIELDS:
                v = values[key][keep]
                row.update({key+'_mean': float(v.mean()), key+'_min': float(v.min()),
                            key+'_max': float(v.max()), key+'_1d': float(matched[key][keep][0]),
                            key+'_ray_rms_difference': float(np.sqrt(np.mean((v-matched[key][keep])**2)))})
            aggregates.append(row)
    for name, records in [('samples.csv', rows), ('radial-summary.csv', aggregates)]:
        with (a.out/name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for ax, key in zip(axes.flat, FIELDS):
        for domain, label in [(0, 'Cartesian'), (1, 'Spherical shell')]:
            selected = [row for row in aggregates if row['domain'] == domain]
            x = np.array([row['r'] for row in selected])
            y = np.array([row[key+'_mean']-row[key+'_1d'] for row in selected])
            line, = ax.plot(x, y, label=label)
            ax.fill_between(x, [row[key+'_min']-row[key+'_1d'] for row in selected],
                            [row[key+'_max']-row[key+'_1d'] for row in selected], color=line.get_color(), alpha=.2)
        ax.axhline(0, color='black', lw=.5)
        ax.set(xlabel='Compactified radius', ylabel='3D minus 1D', title=key)
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(f"Puncture at t/M={3*report['t']:.6g}: 3D n={evolution['n']}, 1D N={report['n']}\nMean and min–max over available fixed rays; domain copies kept separate")
    fig.savefig(a.out/'profiles.png', dpi=180); fig.savefig(a.out/'profiles.pdf')
    result = dict(time=report['t'], time_over_M=3*report['t'], rows=len(data),
                  inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (a.profile, a.evolution_report, a.reference, a.reference_report)},
                  resolutions=dict(n_3d=evolution['n'], n_1d=report['n'], dt_3d=evolution['dt'], dt_1d=report['dt'],
                                   core_layers=evolution['core_evolved_layers'], trace_kappa=evolution['trace_kappa']),
                  max_abs_difference_by_domain=summary,
                  scope='Same-time behavioral comparison, not convergence. Six-point 1D radial interpolation; unweighted available-ray statistics with radius-dependent coverage. No horizon acceptance, constraint norm, or characteristic reflection inference.')
    (a.out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
