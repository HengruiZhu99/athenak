#!/usr/bin/env python3
"""Plot a validated duration export; explicitly distinguish interim evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.data/'summary.json').read_text())
    rows = summary['rows']; times = np.array([row['t'] for row in rows])
    plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix', 'font.size':10, 'axes.labelsize':11,
        'legend.fontsize':9, 'axes.spines.top':False, 'axes.spines.right':False,
        'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.2), layout='constrained')
    a, b, c, d = axes.flat
    for key, symbol, color in zip(['H','M','Theta','Z'],
            [r'\mathcal{H}',r'\mathcal{M}_r',r'\Theta',r'Z_r'],
            ['#0072B2','#D55E00','#009E73','#CC79A7']):
        values = np.array([row[key+'_exterior_rms'] for row in rows])
        assert np.isfinite(values).all()
        a.semilogy(times, np.ma.masked_less_equal(values, 0), color=color,
                   label=f'${symbol}$', lw=1.5)
    a.set(title='(a) Apparent-horizon-exterior constraints',
          ylabel=r'$\|C\|_{r,\,r\geq r_{\rm AH}(t)}$')
    a.legend(ncol=2, loc='lower right')
    mass_error = np.array([row['AH_mass_error']/summary['parameters']['mass'] for row in rows])
    assert np.isfinite(mass_error).all()
    b.semilogy(times, np.ma.masked_less_equal(mass_error, 0), color='#0072B2', lw=1.5)
    b.axhline(1e-5, color='.4', ls='--', lw=.9, label='Declared primary limit')
    b.set(title='(b) Apparent-horizon mass conservation', ylabel=r'$|M_{\rm AH}-M|/M$')
    b.legend(loc='lower right')
    with np.load(args.data/'profiles.npz', allow_pickle=False) as profiles:
        available = summary['profile_times']
        selected = sorted(set([min(available, key=lambda x:abs(x-t))
                               for t in [0, 10, 25, times[-1]]]))
        colors = plt.get_cmap('viridis')(np.linspace(.1, .85, len(selected)))
        for time, color in zip(selected, colors):
            key = f't{round(time*100):06d}'
            r = profiles[key+'_r']; physical = profiles[key+'_physical']
            alpha = profiles[key+'_alpha']
            mask = (r > 0) & (r <= 1e-4)
            assert np.isfinite(physical[mask]).all() and (physical[mask] > 0).all()
            c.loglog(r[mask], physical[mask], color=color, lw=1.4, label=f'$t={time:g}$')
            # r=0 stores an AF coefficient, so show only actual positive-radius samples.
            d.plot(r[1:], alpha[1:], color=color, lw=1.4, label=f'$t={time:g}$')
    floor = summary['parameters']['lapse_floor']
    if floor > 0:
        c.axhline(floor, color='.4', ls='--', lw=.9, label=f'Floor ${floor:.0e}$')
    c.set(title='(c) Physical lapse near the puncture', xlabel='$r$',
          ylabel=r'$\widetilde\alpha=\alpha/\Omega$', xlim=(r[1], 1e-4))
    c.legend(ncol=2, fontsize=8, loc='lower right')
    d.axvline(rows[-1]['AH_r'], color='.55', ls=':', lw=.9,
              label='Last saved horizon')
    d.set(title='(d) Conformal lapse across the domain', xlabel='$r$', ylabel=r'$\alpha$', xlim=(0, 1))
    d.legend(ncol=2, fontsize=8, loc='lower right')
    for ax in [a, b]:
        ax.set(xlabel='$t$', xlim=(0, times[-1]*1.025))
    for ax in axes.flat:
        ax.grid(alpha=.18)
    state = 'Interim' if summary['interim'] else summary['status'].capitalize()
    fig.suptitle(f'{state} attenuation trial: $N={summary["resolution"]}$, saved evidence through $t={times[-1]:g}$', fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output.with_suffix('.'+extension), dpi=210, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
