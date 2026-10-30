#!/usr/bin/env python3
"""Reproduce the N16/N24 comparison using hash-verified saved histories."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'n16-timestep-to3m-v1/n16-original-dt-reproduction-v1'
NEW = OUT.parent / 'n24-at3m'
sources = []

def read(path, artifacts):
    item = next(v for k, v in artifacts.items() if Path(k).name == path.name)
    raw = path.read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    sources.append(dict(path=str(path.relative_to(ROOT)), **item))
    return np.genfromtxt(path, delimiter=',', names=True)

w = json.loads((OLD / 'worker-status.json').read_text())
a = np.concatenate([read(OLD / 'output' / (c['name'] + '.history.csv'), c['artifacts'])
                    for c in w['cases']])
b = read(NEW / 'puncture.history.csv',
         json.loads((NEW / 'local-verification.json').read_text())['artifacts'])

def unique(rows):
    # Merge restart and endpoint-rounding duplicates; retain the latest record.
    latest = {round(float(r['t_over_M']), 10): r for r in rows}
    return np.array([latest[t] for t in sorted(latest) if 0 <= t <= 3], dtype=rows.dtype)

data = {16: unique(a), 24: unique(b)}
colors = {16: '#b45418', 24: '#1769aa'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180})

def setup(title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=17, y=.98)
    fig.text(.5, .935, 'Same log-chi/log-lapse scheme through 3M; dt/dx = 0.04444; core spacing ratio = 1.5',
             ha='center', fontsize=10)
    for ax in axes.flat:
        ax.grid(alpha=.22)
        ax.set_xlim(0, 3)
    for ax in axes[1]: ax.set_xlabel(r'$t/M$')
    fig.subplots_adjust(top=.86, bottom=.18, hspace=.31, wspace=.27)
    return fig, axes

def save(fig, stem):
    for ext in ['png', 'pdf']: fig.savefig(OUT / (stem + '.' + ext), facecolor='white')
    plt.close(fig)

fig, ax = setup('Puncture evolution: N16 and N24 through 3M')
for n, d in data.items():
    t = d['t_over_M']; ok = d['horizon_found'] == 1; c = colors[n]
    candidate = (d['horizon_complete_surface'] == 1) & ~ok
    ax[0, 0].plot(t, np.where(ok, d['horizon_radius'], np.nan), 'o-', color=c, ms=4, label=f'N{n}')
    ax[0, 0].plot(t[candidate], d['horizon_trial_radius'][candidate], 'o', color=c, mfc='white', ms=5)
    err = np.abs(d['horizon_areal_radius'] / (2/3) - 1)
    ax[0, 1].semilogy(t, np.where(ok, err, np.nan), 'o-', color=c, ms=4)
    ax[1, 0].semilogy(t, d['horizon_trial_expansion_rms'], '-', color=c, alpha=.7)
    ax[1, 0].plot(t[ok], d['horizon_trial_expansion_rms'][ok], 'o', color=c, ms=4)
    ax[1, 0].plot(t[candidate], d['horizon_trial_expansion_rms'][candidate], 'o', color=c, mfc='white', ms=5)
    ax[1, 1].plot(t, d['null_surface'] / 1e-15, '-', color=c, lw=2)
    ax[1, 1].plot(t, d['null_ghost'] / 1e-15, ':', color=c, lw=2)
ax[0, 0].set(title='Coordinate horizon radius', ylabel=r'$r_{\rm AH}$ (code units)')
ax[0, 0].legend(loc='upper left')
ax[0, 1].set(title='Accepted horizon areal-radius error', ylabel=r'$|R_{\rm AH}/(2M)-1|$')
ax[1, 0].set(title='Horizon expansion residual', ylabel='RMS expansion')
ax[1, 0].axhline(.001, color='#444444', ls='--', lw=1, label='Acceptance tolerance')
ax[1, 0].legend(fontsize=9, loc='lower right')
ax[1, 1].set(title='Scri null residual (segment running maxima)', ylabel=r'Residual / $10^{-15}$', ylim=(0, 1.9))
ax[1, 1].legend(handles=[Line2D([], [], color='#444444', label='Surface'),
                       Line2D([], [], color='#444444', ls=':', label='Ghost')], fontsize=9)
fig.text(.5, .055, 'Filled circles: accepted horizons. Open circles: complete candidates above tolerance.\n'
         'Missing accepted-horizon diagnostics are left as gaps; no gap interpolation.', ha='center', fontsize=10)
save(fig, 'horizons-and-closure')

fig, ax = setup('Constraint norms: N16 and N24 through 3M')
for n, d in data.items():
    t = d['t_over_M']; c = colors[n]; ok = d['horizon_found'] == 1
    for axis, field in [(ax[0, 0], 'H_L2'), (ax[0, 1], 'Mr_L2'),
                        (ax[1, 0], 'H_exterior_L2'), (ax[1, 1], 'Mr_exterior_L2')]:
        v = d[field] if 'exterior' not in field else np.where(ok, d[field], np.nan)
        axis.plot(t, v, 'o-', color=c, ms=3.5, label=f'N{n}')
ax[0, 0].set(title='Hamiltonian: full domain', ylabel=r'$\|H\|_2$')
ax[0, 1].set(title='Radial momentum: full domain', ylabel=r'$\|M_r\|_2$')
ax[1, 0].set(title='Hamiltonian: outside accepted horizon', ylabel=r'$\|H\|_{2,\mathrm{ext}}$')
ax[1, 1].set(title='Radial momentum: outside accepted horizon', ylabel=r'$\|M_r\|_{2,\mathrm{ext}}$')
ax[0, 0].legend()
fig.text(.5, .055, 'Normalized compactified-coordinate volume norms. Exterior cuts use each run\'s own horizon.\n'
         'Gaps indicate an unaccepted horizon. Two resolutions alone do not establish a convergence order.',
         ha='center', fontsize=10)
save(fig, 'constraints')

summary = {'sources': sources, 'final': {}, 'matching': 'Same-time samples within 1e-8 M; no interpolation.'}
for n, d in data.items():
    f = d[-1]
    summary['final'][n] = {k: float(f[k]) for k in ['t_over_M', 'H_L2', 'Mr_L2',
        'horizon_trial_radius', 'horizon_trial_expansion_rms', 'null_surface', 'null_ghost']}
    summary['final'][n]['accepted_horizon_samples'] = int(sum(d['horizon_found'] == 1))
    summary['final'][n]['samples'] = len(d)
summary['final_reduction_fraction'] = {k: float(1-data[24][-1][k]/data[16][-1][k]) for k in ['H_L2', 'Mr_L2']}
summary['final_candidate_radius_relative_difference'] = float(data[24][-1]['horizon_trial_radius']/data[16][-1]['horizon_trial_radius']-1)
(OUT / 'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
fields = ['H_L2','Mr_L2','H_exterior_L2','Mr_exterior_L2','horizon_found','horizon_trial_radius','horizon_trial_expansion_rms']
rows = []
for r in data[24]:
    i = np.argmin(np.abs(data[16]['t_over_M']-r['t_over_M']))
    if abs(data[16][i]['t_over_M']-r['t_over_M']) < 1e-8:
        rows.append([r['t_over_M']] + [v[k] for v in [data[16][i], r] for k in fields])
np.savetxt(OUT/'matched-times.csv', rows, delimiter=',', comments='',
           header=','.join(['t_over_M']+[f'N{n}_{k}' for n in [16,24] for k in fields]))
print(json.dumps(summary, indent=2))
