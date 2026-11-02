"""Initial-data evidence only: production-grid and bounded radial FD tests."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ns = np.array([16, 24, 32])
grids = [json.loads((HERE / f'n{n}-radial.json').read_text()) for n in ns]
radial = np.genfromtxt(HERE / 'radial-refinement.csv', delimiter=',', names=True)
fixed = np.genfromtxt(HERE / 'radial-fixed2048.csv', delimiter=',', names=True)
coarse = np.genfromtxt(HERE / 'radial-fixed1024.csv', delimiter=',', names=True)

def orders(values, resolutions):
    return (np.log(np.array(values[:-1]) / values[1:]) /
            np.log(np.array(resolutions[1:]) / resolutions[:-1])).tolist()

summary = {
    'scope': 't=0 constraints, not evolved SG1 histories. Native production volume/scri initialization; common exterior cutoff 0.063. Bounded radial test holds angular tensor derivatives analytic and does not exercise patch exchange.',
    'production': {}, 'radial': {}, 'norm_quadrature_control': {},
    'radial_squared_norm_fractions': {}, 'transition_radial_vs_production': {},
}
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
for field, label, color in [('H', 'Hamiltonian', '#b55315'),
                            ('M', 'Momentum magnitude', '#1769aa'),
                            ('Mr', 'Radial momentum', '#228844')]:
    values = [g['norms']['exterior'][field + '_L2'] for g in grids]
    summary['production'][field] = dict(norms=values, effective_orders=orders(values, ns))
    axes[0, 0].loglog(ns, values, 'o-', color=color, label=label)
    vals = fixed[field + '_rms']
    summary['radial'][field] = dict(resolutions=fixed['n'].tolist(), norms=vals.tolist(),
                                   effective_orders=orders(vals, fixed['n']))
    # Refinement of an integration mesh independent of the finite-difference h.
    relative = np.abs(vals / coarse[field + '_rms'] - 1)
    summary['norm_quadrature_control'][field] = {
        'relative_1024_to_2048_change': relative.tolist(),
        'relative_production_radial_centers_to_dense':
            (radial[field + '_rms'] / vals - 1).tolist(),
    }
    if field != 'Mr':
        axes[0, 1].loglog(fixed['n'], vals, 'o-', color=color, label=label + ': fixed 2048-point norm')
        axes[0, 1].loglog(radial['n'], radial[field + '_rms'], 'x--', color=color,
                          label=label + ': actual radial cell centers')
for n, g in zip(ns, grids):
    cumul = g['radial_cumulative']
    ext = next(row for row in cumul if row['cutoff'] == .063)
    summary['radial_squared_norm_fractions'][str(n)] = [
        dict(lo=a['cutoff'], hi=b['cutoff'], **{
            key: (a[key] - b[key]) / ext[key] for key in ['H2', 'M2', 'Mr2']})
        for a, b in zip(cumul, cumul[1:]) if a['cutoff'] >= .063
    ]
    lo = next(row for row in cumul if row['cutoff'] == .45)
    hi = next(row for row in cumul if row['cutoff'] == .75)
    row = radial[radial['n'] == n][0]
    summary['transition_radial_vs_production'][str(n)] = {
        field: dict(production_rms=float(np.sqrt((lo[field+'2']-hi[field+'2']) /
                                                (lo['volume']-hi['volume']))),
                    radial_only_rms=float(row[field+'_rms']),
                    relative_difference=float(row[field+'_rms'] /
                        np.sqrt((lo[field+'2']-hi[field+'2'])/(lo['volume']-hi['volume']))-1))
        for field in ['H', 'M', 'Mr']
    }
for n, color in [(16, '#b55315'), (24, '#1769aa'), (32, '#228844'), (128, '#9050a0')]:
    data = np.genfromtxt(HERE / f'radial-profile-n{n}.csv', delimiter=',', names=True)
    for ax, field in zip(axes[1], ['H', 'M']):
        ax.semilogy(data['r'], np.maximum(np.abs(data[field]), 1e-15), color=color,
                    label=f'N{n} radial FD')
        ax.set(xlim=(.6, .73), xlabel='Compactified radius r', ylabel=f'|{field}|')
axes[0, 0].set(title='Full 3D production-grid constraints at t = 0', xlabel='N', ylabel='Exterior RMS')
axes[0, 0].set_xticks(ns, [str(n) for n in ns], minor=False)
axes[0, 1].set(title='Radial FD4 control: fixed versus moving norm samples', xlabel='N', ylabel='RMS over 0.45 < r < 0.75')
ref = fixed['H_rms'][-1] * (256 / fixed['n']) ** 4
axes[0, 1].loglog(fixed['n'], ref, 'k:', label='Fourth-order reference')
axes[1, 0].set_title('Radial Hamiltonian residual: independent dense samples')
axes[1, 1].set_title('Radial momentum residual: independent dense samples')
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('Native wormhole initial-data audit: underresolved slicing transition', fontsize=15)
fig.tight_layout(rect=(0, .05, 1, .95))
fig.text(.5, .01, 'Radial Hamiltonian approaches fourth order at finer N; full patch/scri consistency is a separate audit.', ha='center', fontsize=10)
fig.savefig(HERE / 'initial-convergence.png', dpi=160)
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:summary[k] for k in ['production', 'radial', 'norm_quadrature_control']}, indent=2))
