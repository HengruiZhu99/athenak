#!/usr/bin/env python3
"""Plot a native scri radiation estimate with its frame and sampling diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scri_news import interpolate_time


def mode20(data):
    labels = [tuple(row) for row in data['ell_m']]
    return data['news_modes'][:, labels.index((2, 0))]


def plot(result, output, controls=()):
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(result/'news.npz')
    summary = json.loads((result/'summary.json').read_text())
    diagnostics = json.loads((result/'diagnostics.json').read_text())
    t = data['bondi_time_over_M0']; news = mode20(data)
    native = data['native_times']/summary['mass_M0']
    rows = 3 if controls else 2
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.0*rows), constrained_layout=True)
    axes[0].plot(t, news.real, color='#175d91', label=r'$\mathrm{Re}\,N_{20}$')
    axes[0].plot(t, news.imag, color='#aa493e', linestyle='--', label=r'$\mathrm{Im}\,N_{20}$')
    axes[0].set(xlabel=r'Transported time $u/M_0$', ylabel=r'Formal news projection $N_{20}$')
    axes[0].legend(loc='best', frameon=False)
    for key, label, color in [('round_cut_rms', 'Round-cut defect (RMS)', '#175d91'),
                              ('round_shear_max', 'Round-cut shear (max)', '#bf7824'),
                              ('clock_transport_max', 'Clock transport residual (max)', '#666666')]:
        values = np.array([d[key] for d in diagnostics])
        axes[1].plot(native, np.where(values > 0, values, np.nan), label=label, color=color)
    axes[1].set(yscale='log', xlabel=r'Native time $t/M_0$', ylabel='Dimensionless defect')
    axes[1].legend(loc='best', frameon=False)
    comparisons = []
    for path in controls:
        other = np.load(path/'news.npz'); other_summary = json.loads((path/'summary.json').read_text())
        u = other['bondi_time_over_M0']; values = mode20(other)
        mask = (t >= u[0]) & (t <= u[-1]); common = t[mask]
        if not len(common):
            raise ValueError('control has no common transported time interval')
        delta = interpolate_time(u, values, common)-news[mask]
        peak = float(np.max(abs(news[mask])))
        label = f"L={other_summary['angular_lmax']}, sample stride={other_summary['temporal_stride']}"
        axes[2].plot(common, abs(delta), label=label)
        comparisons.append(dict(label=label, result=str(path),
            max_abs_difference=float(np.max(abs(delta))),
            rms_difference=float(np.sqrt(np.mean(abs(delta)**2))),
            max_difference_over_peak=float(np.max(abs(delta))/peak) if peak else None,
            common_time_range_M0=[float(common[0]), float(common[-1])]))
    if controls:
        axes[2].set(xlabel=r'Transported time $u/M_0$', ylabel=r'$|\Delta N_{20}|$')
        axes[2].legend(loc='best', frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.tick_params(direction='in')
    fig.suptitle('Native scri radiation diagnostic\nPhysical waveform not validated', fontsize=14)
    fig.savefig(output/'news_diagnostic.png', dpi=180)
    fig.savefig(output/'news_diagnostic.pdf')
    plt.close(fig)
    np.savetxt(output/'news20.csv', np.column_stack((t, news.real, news.imag)), delimiter=',',
               header='transported_u_over_M0,formal_N20_real,formal_N20_imag', comments='')
    receipt = dict(result=str(result), news_sha256=hashlib.sha256((result/'news.npz').read_bytes()).hexdigest(),
                   physical_waveform_validated=False, comparisons=comparisons)
    (output/'plot-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    plot(args.result.resolve(), args.output.resolve(), [p.resolve() for p in args.controls])
