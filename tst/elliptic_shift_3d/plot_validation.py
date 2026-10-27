#!/usr/bin/env python3
"""Standalone scientific figures from retained, hash-bound validation records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/elliptic-shift-3d-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    root = args.evidence
    manifest = json.loads((root/'manifest.json').read_text())
    tests = manifest['tests']
    audit_path = root/'surface-interpolation-audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for curved, label in [(0, 'flat metric'), (1, 'curved metric (nonzero Cotton)')]:
        rows = [tests[f'mms-curve{curved}-n{n}']['result'] for n in (8, 12, 16)]
        n = np.array([r['n'] for r in rows])
        error = np.array([r['error_rms'] for r in rows])
        ax[0, 0].loglog(n, error, 'o-', label=label)
        ax[0, 1].loglog(n, [r['truncation_rms'] for r in rows], 'o-', label=label)
        ax[1, 1].plot([r['cells'] for r in rows], [r['solve_seconds'] for r in rows], 'o-', label=label)
        if curved:
            for row in rows:
                history = row['residual_history']
                iteration = np.minimum(np.arange(len(history))*24, row['iterations'])
                ax[1, 0].semilogy(iteration, history, 'o-', label=f"N={row['n']}")
            ax[0, 0].loglog(n, error[0]*(n[0]/n)**4, 'k--', label=r'$N^{-4}$')
    for a in ax.flat:
        a.legend(fontsize=8)
    ax[0, 0].set(xlabel='active cells per block axis N', ylabel='solution RMS error', title='Vector manufactured solutions')
    ax[0, 1].set(xlabel='N', ylabel='continuum-to-discrete residual RMS', title='Discretization error vs solver error')
    ax[1, 0].set(xlabel='Krylov iterations (true residual checks)', ylabel='original vector residual RMS', title='Variable metric, all patch interfaces coupled')
    ax[1, 1].set(xlabel='active cells', ylabel='local CPU solve seconds', title='Small-run cost; no scalability claim')
    fig.savefig(root/'elliptic-convergence.png', dpi=170)
    plt.close(fig)

    reference = np.load(root/'radial-reference.npz')
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    ax[0, 0].plot(reference['r'], reference['alpha']-(1+reference['r']**2)/2,
                  'k-', label='independent 1D, N=513')
    ax[0, 1].plot(reference['r'], reference['A_rr'], 'k-', label='independent 1D')
    for n in (6, 8, 12):
        rows = manifest['matched_sphere_comparisons'][str(n)]
        radius = np.array([r['r'] for r in rows])
        for field, top, bottom in [('alpha', ax[0, 0], ax[1, 0]), ('A_rr', ax[0, 1], ax[1, 1])]:
            mean = np.array([r[field]['mean'] for r in rows])
            scatter = np.array([r[field]['angular_scatter'] for r in rows])
            exact = np.array([r[field]['reference'] for r in rows])
            plotted = mean-(1+radius**2)/2 if field == 'alpha' else mean
            top.plot(radius, plotted, 'o', markersize=3, label=f'3D N={n}, mean of 64 angles')
            difference = mean-exact
            if audit:
                controlled = audit['cases'][str(n)]['rows']
                difference = np.array([r[field]['mean_error'] for r in controlled])
                scatter = np.array([r[field]['angular_scatter'] for r in controlled])
            line = bottom.plot(radius, difference, 'o-', markersize=3, label=f'N={n}')[0]
            bottom.fill_between(radius, difference-scatter, difference+scatter,
                                color=line.get_color(), alpha=.18)
    ax[0, 0].set(ylabel='α − αref', title='Radial lapse pulse, 3D evolution at t=0.002')
    ax[0, 1].set(ylabel='conformal A_rr', title='Independent 1D comparison; free 3D evolution')
    ax[1, 0].set(ylabel='lapse mean error ± angular scatter', title='Matched extraction interpolation control')
    ax[1, 1].set(ylabel='curvature mean error ± angular scatter', title='64 angles per radius; raw errors also retained')
    for a in ax.flat:
        a.title.set_fontsize(10)
        a.set_xlabel('compact radius r')
        a.legend(fontsize=7)
        a.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    fig.savefig(root/'radial-3d-comparison.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for prefix, label in [('radial', 'radial, one shell'), ('angular-two-shell', 'nonspherical, two shells')]:
        rows = [tests[f'{prefix}-n{n}']['result'] for n in (6, 8, 12)]
        ax[0].semilogy([6, 8, 12], [r['scri_clock_extrapolation'] for r in rows], 'o-', label=label)
        ax[1].semilogy([6, 8, 12], [r['scri_null_extrapolation'] for r in rows], 'o-', label=label)
    ax[0].axhline(1e-13, color='k', linestyle='--', label='original spherical clock target')
    ax[1].axhline(1e-12, color='k', linestyle='--', label='original spherical null target')
    ax[0].set(ylabel='independent active-only face clock defect', title='Prescribed face data vs interior consistency')
    ax[1].set(ylabel='independent face future-null defect', title='Provisional short pilots; failures retained')
    for a in ax:
        a.title.set_fontsize(10)
        a.set_xlabel('active cells per block axis N')
        a.legend(fontsize=7)
    fig.savefig(root/'coupled-boundary-consistency.png', dpi=170)
    plt.close(fig)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.png')}
    (root/'figure-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(hashes, indent=2))


if __name__ == '__main__':
    main()
