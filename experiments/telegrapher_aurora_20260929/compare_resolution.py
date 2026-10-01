#!/usr/bin/env python3
"""Compare overlapping gamma-five runs, retaining actual sample time offsets."""
import argparse
import json
import math
from pathlib import Path
from analyze import analyze, numeric_rows
from diagnostics import slice_peaks


def parameters(run):
    blocks, name = {}, None
    for line in (run / 'input.athinput').read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<') and line.endswith('>'):
            name = line[1:-1]
            blocks.setdefault(name, {})
        elif '=' in line and name:
            key, value = line.split('=', 1)
            blocks[name][key.strip()] = value.strip()
    return blocks


def compare(coarse, fine):
    decks = [parameters(run) for run in (coarse, fine)]
    required = {'problem': ['punc_ADM_mass', 'punc_velocity_x1', 'pgen_name'],
                'z4c': ['telegraph_lapse', 'telegraph_tau', 'telegraph_kappa',
                        'damp_kappa1', 'damp_kappa2', 'co_0_radius'],
                'time': ['integrator', 'cfl_number'],
                'mesh': ['nx1', 'nx2', 'nx3', 'x1min', 'x1max', 'x2min',
                         'x2max', 'x3min', 'x3max']}
    matched = {}
    for block, keys in required.items():
        for key in keys:
            a, b = (deck[block][key] for deck in decks)
            if a != b:
                raise ValueError('Different physical setting: ' + block + '/' + key)
            matched[block + '/' + key] = a
    mass = float(decks[0]['problem']['punc_ADM_mass'])
    velocity = float(decks[0]['problem']['punc_velocity_x1'])
    if abs(mass - 1) > 1e-12 or abs(1 / math.sqrt(1 - velocity**2) - 5) > 1e-8:
        raise ValueError('This comparison expects rest mass 1 and gamma 5')
    masks = [float(deck['z4c'].get('excise_chi', '.0625')) for deck in decks]
    if masks != [.0625, .0625]:
        raise ValueError('This comparison expects chi>=0.0625 for both runs')
    matched['z4c/excise_chi'] = '.0625 (explicit or code default)'
    spacings = []
    for deck in decks:
        mesh = deck['mesh']
        spacings.append((float(mesh['x1max']) - float(mesh['x1min'])) /
                        (int(mesh['nx1']) * 2**int(deck['z4c']['co_0_reflevel'])))
    if abs(spacings[0] / spacings[1] - 2) > 1e-12:
        raise ValueError('Expected factor-two finest-grid resolution difference')
    histories = [numeric_rows(next(run.glob('*.hst'))) for run in (coarse, fine)]
    overlap_end = min(rows[-1][0] for rows in histories)
    norms = []
    for t in [0, .1, 1, 2, 3, 4, 4.5, 5, 10, 15, 20]:
        if t > overlap_end:
            continue
        a, b = (min(rows, key=lambda r: abs(r[0] - t)) for rows in histories)
        if max(abs(a[0] - t), abs(b[0] - t)) > .06:
            continue
        norms.append(dict(target_time=t, coarse_time=a[0], fine_time=b[0],
                          fine_minus_coarse_time=b[0] - a[0],
                          coarse_H_norm2=a[3], fine_H_norm2=b[3],
                          fine_over_coarse_H_norm2=b[3] / a[3] if a[3] else None))
    peaks = [slice_peaks(run) for run in (coarse, fine)]
    slices = []
    for t in range(int(overlap_end) + 1):
        a, b = (min(rows, key=lambda r: abs(r['time'] - t)) for rows in peaks)
        if max(abs(a['time'] - t), abs(b['time'] - t)) > .01:
            continue
        slices.append(dict(target_time=t, coarse_peak=a, fine_peak=b,
                           coarse_over_fine_abs_H=a['max_abs_H'] / b['max_abs_H']))
    return dict(coarse=analyze(coarse), fine=analyze(fine), finest_dx=spacings,
                matching_physical_parameters=matched, overlap_end=overlap_end,
                integrated_norm_comparison=norms, x_axis_peak_comparison=slices,
                note='Partial comparison when either run is unfinished. Chi>=0.0625 '
                     'is a coordinate mask, not horizon excision. Adding an AMR level '
                     'does not uniformly refine the entire domain. The slice maxima '
                     'can occur at different locations; no convergence order is fitted.')


def plot_comparison(data, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    norms, peaks = data['integrated_norm_comparison'], data['x_axis_peak_comparison']
    for name, color, spacing in zip(('coarse', 'fine'), ('#c44e52', '#4c72b0'),
                                    data['finest_dx']):
        label = 'Finest dx = 1/' + str(round(1 / spacing))
        axes[0].semilogy([r[name + '_time'] for r in norms],
                         [r[name + '_H_norm2'] for r in norms],
                         'o-', color=color, label=label)
        axes[1].semilogy([r[name + '_peak']['time'] for r in peaks],
                         [r[name + '_peak']['max_abs_H'] for r in peaks],
                         'o-', color=color, label=label)
    axes[0].set_ylabel(r'$\int_{\chi\geq0.0625} H^2\,dV$')
    axes[1].set_ylabel(r'Largest sampled x-axis $|H|$, $\chi\geq0.0625$')
    for ax in axes:
        ax.set_xlabel('Time / rest mass')
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    qualifier = 'partial ' if not all(data[key]['completed_20M']
                                    for key in ('coarse', 'fine')) else ''
    fig.suptitle(f'Gamma = 5: {qualifier}comparison through {data["overlap_end"]:.1f} M\n'
                 'An added AMR level refines the central region; the chi mask retains interior points',
                 fontsize=10)
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(stem) + suffix, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coarse', type=Path)
    parser.add_argument('fine', type=Path)
    parser.add_argument('--plot', type=Path, help='Optional PNG/PDF output stem')
    args = parser.parse_args()
    data = compare(args.coarse, args.fine)
    if args.plot:
        plot_comparison(data, args.plot)
    print(json.dumps(data, indent=2, allow_nan=False))
