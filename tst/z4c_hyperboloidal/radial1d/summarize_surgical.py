#!/usr/bin/env python3
"""Summarize completed radial experiments on fixed, common time windows."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return np.genfromtxt(path/'scri.csv', names=True, delimiter=',')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    root = a.output.resolve()
    result = {}
    for receipt in sorted(root.glob('*/receipt.json')):
        r = json.loads(receipt.read_text())
        d = read(receipt.parent)
        if r['returncode'] != 0 or not all(np.isfinite(d[k]).all() for k in d.dtype.names):
            raise ValueError('Failed or nonfinite run: '+receipt.parent.name)
        row = dict(returncode=r['returncode'], seconds=r['seconds'],
                   final_time=float(d['time'][-1]), windows={})
        for end in [.8, 1.6]:
            if d['time'][-1] < end:
                continue
            z = d[d['time'] <= end]
            row['windows'][str(end)] = {
                'peak_T': float(np.max(np.abs(z['T_max']))),
                'peak_clock': float(np.max(np.abs(z['clock']))),
                'final_T': float(z['T_max'][-1])}
        result[receipt.parent.name] = row
    # These are unlifted profile constraints. Restrict to an interior annulus
    # where the native normal lift is identically zero at every tested N.
    result['interior_constraints'] = {}
    for tag in ['base', 'damping']:
        for n in [12, 20, 32, 64]:
            path = root/f'long-{tag}-N{n}'/'profile-003.csv'
            d = np.genfromtxt(path, names=True, delimiter=',', skip_header=1)
            use = (d['r'] >= .7) & (d['r'] <= .95)
            result['interior_constraints'][f'{tag}-N{n}'] = {
                'sample_time': float(path.read_text().splitlines()[0].split('=')[1]),
                'max_H': float(np.max(np.abs(d['hamiltonian'][use]))),
                'max_Mr': float(np.max(np.abs(d['momentum'][use])))}
    # Full signed tensor components are compared, not differences of maxima.
    result['halfdt_checks'] = {}
    for prefix, n in [('control', 20), ('wide', 16)]:
        coarse = read(root/f'{prefix}-damping-N{n}')
        fine = read(root/f'{prefix}-damping-halfdt-N{n}')
        coarse = coarse[coarse['time'] <= .8]
        check = {}
        for field in ['clock', 'T00', 'T0x', 'Txx', 'Tyy', 'Tzz']:
            diff = coarse[field]-np.interp(coarse['time'], fine['time'], fine[field])
            check[field] = float(np.max(np.abs(diff)))
        result['halfdt_checks'][prefix] = check
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for tag, label in [('base', r'$\kappa_\alpha=0.25$'),
                       ('damping', r'$\kappa_\alpha=0.5$')]:
        for ax, family, ns, end in [(axes[0], 'control', [12, 16, 20, 32, 64], '.8'),
                                     (axes[1], 'wide', [16, 32, 64], '.8'),
                                     (axes[2], 'long', [12, 20, 32, 64], '1.6')]:
            vals = []
            for n in ns:
                key = f'{family}-{tag}-N{n}'
                if key not in result and family == 'control':
                    key = f'long-{tag}-N{n}'
                vals.append(result[key]['windows'][str(float(end))]['peak_T'])
            ax.loglog(ns, vals, 'o-', label=label)
    axes[0].loglog([12, 16], [result[f'control-endpoint9-N{n}']['windows']['0.8']['peak_T']
                              for n in [12, 16]], 'x--', label='9-point endpoint, 0.25')
    if 'control-moderate-N64' in result:
        ns = [12, 16, 20, 32, 64]
        axes[0].loglog(ns, [result[f'control-moderate-N{n}']['windows']['0.8']['peak_T']
                            for n in ns], 's:', label=r'$\kappa_\alpha=0.375$')
    if 'control-damping-N128' in result:
        axes[0].loglog([64, 128], [result[k]['windows']['0.8']['peak_T'] for k in
                      ['long-damping-N64', 'control-damping-N128']], 'o-', color='C1')
    if 'control-boost-N64' in result:
        ns = [12, 16, 20, 32, 64]
        axes[0].loglog(ns, [result[f'control-boost-N{n}']['windows']['0.8']['peak_T']
                            for n in ns], '+--', label='Compact boost, outer 0.25')
    for ax, title in zip(axes, ['Control: 0–2.4 M', 'Wide pulse: 0–2.4 M',
                               'Control: 0–4.8 M']):
        ax.set_title(title)
        ax.set_xlabel('Equivalent outer-shell resolution N')
        ax.set_ylabel(r'Peak max $|T_{ab}|$ at scri')
        ax.grid(True, which='both', alpha=.2)
    for ax, ticks in zip(axes, [[12, 16, 20, 32, 64, 128], [16, 32, 64],
                               [12, 20, 32, 64]]):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=7)
    fig.suptitle('Radial surrogate — no angular/seam evolution or physical-news test')
    fig.savefig(root/'comparison.png', dpi=180)
    fig.savefig(root/'comparison.pdf')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
