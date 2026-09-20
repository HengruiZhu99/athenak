#!/usr/bin/env python3
"""Plot validated Theta profiles; NPZ/manifest must come from aggregate.py."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm


def time_edges(time):
    if len(time) == 1:
        return np.array([0., max(64., time[0]+32.)])
    return np.r_[max(0., time[0]-(time[1]-time[0])/2), (time[:-1]+time[1:])/2,
                 time[-1]+(time[-1]-time[-2])/2]


def main(root):
    cases = [('theta_primary', r'$\eta=0.02,\ d_\alpha=0.01$'),
             ('theta_lapse01', r'$\eta=0.02,\ d_\alpha=0.1$'),
             ('theta_amplitude', r'$A=10^{-7},\ \eta=0.02,\ d_\alpha=0.01$')]
    existing = [(case, label) for case, label in cases if (root/case/'profiles.npz').exists()]
    if not existing:
        return
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'font.size': 10, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(2, len(existing), figsize=(6*len(existing), 7), squeeze=False,
                           layout='constrained')
    trace, tax = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    summary = {'weighting': 'coordinate volume dx^3; not proper volume', 'cases': {}}
    for col, (case, label) in enumerate(existing):
        d = dict(np.load(root/case/'profiles.npz'))
        manifest = json.loads((root/case/'manifest.json').read_text())
        t, edges, te = d['time'], d['edges'], time_edges(d['time'])
        absent = d['count'].T == 0
        signed = np.ma.array(d['mean'].T, mask=absent)
        rms = np.ma.array(d['rms'].T, mask=absent | (d['rms'].T == 0))
        im0 = ax[0,col].pcolormesh(te, edges, signed, shading='flat', cmap='RdBu_r',
                    norm=SymLogNorm(linthresh=1e-13, vmin=-1e-6, vmax=1e-6, base=10), rasterized=True)
        im1 = ax[1,col].pcolormesh(te, edges, rms, shading='flat', cmap='magma',
                    norm=LogNorm(vmin=1e-15, vmax=1e-6), rasterized=True)
        for row in (0, 1):
            for r, style in ((512, '--'), (1792, '--'), (2048, ':')):
                ax[row,col].axhline(r, color='.4', ls=style, lw=.8)
            ax[row,col].set(ylabel=r'$r/M$', xlim=(0, t[-1]), ylim=(0, 3548))
        ax[0,col].set_title(label + '\n' + 'signed radial mean')
        ax[1,col].set_title('radial RMS')
        ax[1,col].set_xlabel(r'$t/M$')
        fig.colorbar(im0, ax=ax[0,col], label=r'$\langle\Theta\rangle_r$')
        fig.colorbar(im1, ax=ax[1,col], label=r'$\sqrt{\langle\Theta^2\rangle_r}$')
        # Region traces are RMS, not an estimate of incoming/outgoing flux.
        item = {'snapshots': len(t), 'latest_time': float(t[-1]), 'regions': {},
                'latest_active_peak': manifest['included'][-1]['peak']}
        for n, name in enumerate(['core r≤512M', 'ramp 512–1792M', 'plateau r≥1792M']):
            v = d['regional_rms'][:,n]
            tax[n].semilogy(t, v, label=label)
            tax[n].set(title=name, xlabel=r'$t/M$', ylabel=r'coordinate-volume $\Theta$ RMS')
            tax[n].grid(alpha=.2)
            imin = np.argmin(v)
            data = {'initial_RMS': float(v[0]), 'latest_RMS': float(v[-1]),
                    'latest_to_initial': float(v[-1]/v[0]),
                    'minimum_RMS': float(v[imin]), 'time_of_minimum': float(t[imin]),
                    'latest_over_minimum': float(v[-1]/v[imin]),
                    'latest_max_abs': float(d['regional_maxabs'][-1,n])}
            for ta, tb in [(2000, 5000), (5000, 10000), (10000, 20000), (20000, 35000), (35000, 50000)]:
                mask = (t >= ta) & (t <= tb) & (v > 0)
                if mask.sum() >= 8:
                    p = np.polyfit(t[mask], np.log(v[mask]), 1)
                    data['fit_%d_%d' % (ta,tb)] = {'gamma': float(p[0]),
                        'actual_first_time': float(t[mask][0]), 'actual_last_time': float(t[mask][-1]),
                        'scope': 'log-linear descriptive fit, not proof of a pure eigenmode'}
            item['regions'][name] = data
        summary['cases'][case] = item
    tax[0].legend(frameon=False, fontsize=9)
    fig.savefig(root/'theta-radial-spacetime.png', dpi=170)
    fig.savefig(root/'theta-radial-spacetime.pdf')
    trace.savefig(root/'theta-regional-rms.png', dpi=170)
    trace.savefig(root/'theta-regional-rms.pdf')
    plt.close('all')
    (root/'propagation-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path, nargs='?', default=Path(__file__).resolve().parent)
    main(p.parse_args().directory)
