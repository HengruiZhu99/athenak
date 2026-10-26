#!/usr/bin/env python3
"""Additional read-only checks and a focused pulse-resolution illustration."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.signal import find_peaks, peak_widths

import localize_schwarzschild_error as loc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    E, old, out = args.evidence.resolve(), args.initial.resolve(), args.output.resolve()
    if out.exists():
        raise RuntimeError('Preserve earlier diagnostics; select a fresh output')
    out.mkdir(parents=True)
    r = loc.read(old/'review.json')
    # Bind the numerical localization code used for the parent diagnostic.
    if r['source_sha256'][str(Path(loc.__file__).resolve())] != loc.sha(loc.__file__):
        raise RuntimeError('Parent analysis source has changed')
    source = loc.read(E/'cpu-schwarzschild-commensurate-finer/primary-resolved-reference-review.json')
    runs, ns = [Path(x) for x in source['runs']], r['resolutions']
    O = E/'radial-oracle/smooth'
    a = np.load(old/'plot-data.npz')
    early = {}
    for time in [2, 6, 12]:
        refs = [loc.profile(O, n, time) for n in [2400, 4800]]
        early[time] = {}
        for n, run in zip(ns, runs):
            print(f'early regional check t={time}, N={n}', flush=True)
            data = loc.snapshot(run, time)
            regional, _, _ = loc.stats(data, refs, a['bin_edges'])
            early[time][n] = regional
            del data
    early_orders = {t: {reg: {key: loc.orders([early[t][n][reg]['errors'][key]['L2'] for n in ns], ns)
                              for key in ['psi', 'Dpsi', 'reduction']}
                        for reg in loc.REGIONS+['all exterior', 'excluded inner region']}
                    for t in early}

    ref = loc.profile(O, 4800, 12)
    rho, P = ref['rho'], ref['psi']
    derivative = np.gradient(P, rho)
    band = (rho >= 8) & (rho <= 16)
    indices = np.where(band)[0]
    peaks, _ = find_peaks(abs(derivative[band]))
    k = peaks[np.argmax(abs(derivative[band][peaks]))]
    widths = peak_widths(abs(derivative[band]), [k], rel_height=.5)
    left = np.interp(widths[2][0], np.arange(len(indices)), rho[band])
    right = np.interp(widths[3][0], np.arange(len(indices)), rho[band])
    resolution = dict(time=12, direction=[.8, .6, 0], gradient_peak_radius=float(rho[indices[k]]),
                      gradient_peak_half_prominence_interval=[float(left), float(right)], width=float(right-left),
                      shell_radial_nodes_in_interval={n: int(np.sum((a[f'ray_nodes_n{n}']>=left)&(a[f'ray_nodes_n{n}']<=right))) for n in ns},
                      interpretation='Feature sampling diagnostic, not a universal points-per-wavelength admission rule; no causal attribution from a single snapshot.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for n, color in zip(ns, loc.COLORS):
        nodes = a[f'ray_nodes_n{n}']; values = a[f'ray_node_psi_t12_n{n}']
        axes[0, 0].plot(nodes, values, 'o', ms=3, color=color, label=f'N={n}: {resolution["shell_radial_nodes_in_interval"][n]} nodes in shaded width')
        axes[1, 0].plot(a['ray_rho'], a[f'ray_psi_t12_n{n}']-a['ray_reference_t12'], color=color, label=f'N={n}')
    axes[0, 0].plot(rho, P, 'k-', lw=1, label='independent 4800')
    axes[0, 0].set(xlim=(9, 16), ylabel='radial amplitude psi/(nx ny)', title='Outgoing feature in the transition at 12M')
    axes[0, 0].axvspan(left, right, color='gray', alpha=.15);axes[0, 0].legend(fontsize=8)
    axes[1, 0].set(xlim=(9, 16), xlabel='compact radius rho/M', ylabel='signed radial-amplitude error');axes[1, 0].legend(fontsize=8)
    axes[0, 1].plot(rho, derivative, 'k-')
    axes[0, 1].axvspan(left, right, color='gray', alpha=.15)
    axes[0, 1].set(xlim=(9, 16), ylabel='reference radial derivative', title=f'Steep-gradient width: {right-left:.2f}M (half prominence)')
    x=np.arange(len(loc.REGIONS))
    for time, color in zip([2, 6, 12, 100], ['#cc79a7', '#e69f00', '#56b4e9', '#000000']):
        values = [early_orders[time][reg]['psi'][1] if time != 100 else r['regional_orders'][reg]['psi']['L2'][1] for reg in loc.REGIONS]
        axes[1, 1].plot(x, values, 'o-', color=color, label=f't={time}M')
    axes[1, 1].axhline(1.5, color='black', ls='--')
    axes[1, 1].set(xticks=x, xticklabels=loc.REGIONS, ylabel='regional field RMS order N48/60', title='Evolution of the regional convergence pattern')
    axes[1, 1].tick_params(axis='x', labelrotation=25);axes[1, 1].legend(fontsize=8)
    fig.suptitle('Transition resolution: actual radial nodes along direction (0.8,0.6,0)\nThe late error location does not by itself establish where that error was generated', fontsize=12)
    loc.save(fig, out, 'transition-resolution')

    # Clearer scri plot: focus on the non-negligible outgoing signal samples.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for time, color in zip([6, 12, 100], ['#e69f00', '#56b4e9', '#000000']):
        for ax, key, norm, label in [
                (axes[0, 0], 'psi', 'mode_error', 'radiation (2,2) mode absolute error'),
                (axes[0, 1], 'psi', 'angular_L2', 'endpoint angular field RMS error'),
                (axes[1, 0], 'v', 'mode_error', 'radiation time-derivative mode error')]:
            vals = [r['scri'][str(time)][str(n)]['fields'][key][norm] for n in ns]
            order = r['scri_orders'][str(time)][key][norm][1]
            ax.loglog(ns, vals, 'o-', color=color, label=f't={time}M; p48/60={order:.2f}')
            ax.set(xlabel='resolution N', ylabel=label, xticks=ns)
            ax.set_xticklabels(ns);ax.xaxis.set_minor_locator(NullLocator());ax.legend(fontsize=9);ax.grid(alpha=.2)
    ax=axes[1, 1]
    for n, color in zip(ns, loc.COLORS):
        frac = [r['scri'][str(t)][str(n)]['fields']['psi']['mode_error']/abs(r['scri'][str(t)][str(n)]['fields']['psi']['reference_radiation_mode_imag']) for t in [6, 12, 100]]
        ax.semilogy([6,12,100], frac, 'o', color=color, label=f'N={n}')
    ax.set(xlabel='saved time t/M', ylabel='relative radiation-mode amplitude error', title='Samples only; no connecting time interpolation');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Direct convergence at exact scri: radiation r Phi = 24 psi\nKnown-mode projection is normalized for angular quadrature; t=2 precursor is negligible and omitted', fontsize=12)
    loc.save(fig, out, 'scri-signal-convergence')

    # Machine-readable tables of the full late localization and direct scri checks.
    with (out/'late-regional-errors.csv').open('w') as stream:
        import csv
        writer=csv.writer(stream);writer.writerow(['region','field','N','RMS','Linf','squared_error_fraction','p36_48_RMS','p48_60_RMS','reference_fraction_RMS'])
        for region in loc.REGIONS+['all exterior','excluded inner region']:
            for key in loc.FIELDS:
                for n in ns:
                    d=r['regional'][str(n)][region]['errors'][key]
                    writer.writerow([region,key,n,d['L2'],d['Linf'],d.get('squared_error_fraction'),*r['regional_orders'][region][key]['L2'],d.get('reference_fraction_L2')])
    with (out/'scri-snapshot-convergence.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['time','field','N','mode_imag','reference_mode_imag','mode_error','angular_RMS','p36_48_mode','p48_60_mode','reference_mode_difference','raw_quadrature_bias'])
        for time in r['times']:
            for key in ['psi','v']:
                for n in ns:
                    d=r['scri'][str(time)][str(n)]['fields'][key]
                    writer.writerow([time,key,n,d['radiation_mode_imag'],d['reference_radiation_mode_imag'],d['mode_error'],d['angular_L2'],*r['scri_orders'][str(time)][key]['mode_error'],d['reference_mode_difference'],d['raw_quadrature_bias']])
    for source in [Path(loc.__file__), Path(__file__), Path(loc.__file__).with_name('reference_interpolation.py')]:
        shutil.copy2(source, out/source.name)
    result = dict(parent_diagnostic=str(old/'review.json'), parent_sha256=loc.sha(old/'review.json'),
                  transient_regional_errors=early, transient_regional_RMS_orders=early_orders,
                  pulse_resolution=resolution, additional_trajectories=0,
                  limits=['Spatial location of residual error is not a proof of its causal origin.',
                          'No dense scri time series was saved; direct waveform checks are at 2,6,12,100M only.',
                          'Early scri mode convergence fails 1.5 on N48/60 at6/12; no full scri waveform admission.',
                          'Quarter/half timestep controls and dyadic N96 late comparison remain separate required evidence.'],
                  input_sha256={str(path):loc.sha(path) for path in sorted(loc.INPUTS)},
                  output_sha256={path.name:loc.sha(path) for path in out.iterdir() if path.is_file()})
    (out/'review.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(pulse_resolution=resolution, early_orders=early_orders), indent=2), flush=True)


if __name__ == '__main__':
    main()
