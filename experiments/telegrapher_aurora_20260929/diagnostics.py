#!/usr/bin/env python3
"""Plot saved campaign data and locate Hamiltonian peaks on the x-axis slice."""
import argparse
import gzip
import json
import math
import re
from pathlib import Path

from analyze import analyze, numeric_rows


def slice_data(path):
    text = (gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz'
            else path.read_text())
    lines = text.splitlines()
    time = float(re.search(r'time=([\d.eE+-]+)', lines[0])[1])
    names = lines[1].split()[1:]
    return time, [dict(zip(names, row)) for row in numeric_rows(path)]


def slice_peaks(run):
    trackers = list(run.glob('*.co_0.txt'))
    tracker = numeric_rows(trackers[0]) if trackers else []
    peaks = []
    for path in sorted((run / 'tab').glob('*.con.*.tab*')):
        time, rows = slice_data(path)
        zpath = path.with_name(path.name.replace('.con.', '.z4c.'))
        if not zpath.exists():
            continue
        _, zrows = slice_data(zpath)
        chi = {(r['gid'], r['i']): r['z4c_chi'] for r in zrows}
        finite = [r for r in rows if math.isfinite(r['con_H']) and
                  chi.get((r['gid'], r['i']), 0) >= 0.0625]
        if not finite:
            continue
        peak = max(finite, key=lambda r: abs(r['con_H']))
        center = min(tracker, key=lambda r: abs(r[1] - time))[2] if tracker else 0
        peaks.append(dict(time=time, max_abs_H=abs(peak['con_H']),
                          x=peak['x1v'], tracker_x=center,
                          x_minus_tracker=peak['x1v'] - center,
                          chi=chi[(peak['gid'], peak['i'])],
                          gid=int(peak['gid']), cell_i=int(peak['i']),
                          cells_from_block_edge=int(min(peak['i'] - 4,
                                                        35 - peak['i']))))
    return peaks


def peak_horizon_comparison(run, peaks, outcome):
    """Compare the slice peak with the nearest qualified horizon sample.

    Times are not identical; retain the offset rather than claiming synchronous
    horizon excision or using this comparison to change the history norm.
    """
    if not peaks or not outcome['qualified_tracking_20M']:
        return None
    peak = max(peaks, key=lambda row: row['max_abs_H'])
    summaries = list(run.glob('*horizon_summary_0.txt'))
    if summaries:
        rows = [r for r in numeric_rows(summaries[0]) if any(
            abs(r[1] - t) < 5e-4 for t in outcome['successful_horizon_times'])]
    else:
        summaries = list((run / 'horizon').glob('BHaHAHA_diagnostics.ah1.gp'))
        rows = numeric_rows(summaries[0]) if summaries else []
    if not rows:
        return None
    ah = min(rows, key=lambda row: abs(row[1] - peak['time']))
    if outcome['finder'] == 'fastflow':
        tracker = numeric_rows(next(run.glob('*.co_0.txt')))
        center = min(tracker, key=lambda row: abs(row[1] - ah[1]))[2:5]
        radius = ah[11]
        center_source = 'nearest tracker sample (fastflow grid center)'
    else:
        center = ah[2:5]
        radius = ah[5]
        center_source = 'BHaHAHA area centroid'
    distance = math.sqrt((peak['x'] - center[0])**2 + center[1]**2 + center[2]**2)
    return dict(slice_peak=peak, nearest_horizon_time=ah[1],
                horizon_time_offset=ah[1] - peak['time'],
                center_source=center_source,
                nearest_horizon_min_radius=radius,
                distance_from_nearest_horizon_center=distance,
                inside_nearest_horizon_min_radius=distance < radius)


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = sorted(p.parent for p in root.rglob('run.log')
                  if 'evolution_' in str(p.parent))
    if not runs:
        raise SystemExit('No evolution data under ' + str(root))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             constrained_layout=True)
    peaks = {}
    peak_comparisons = {}
    for run in runs:
        outcome = analyze(run)
        job = run.parent.name.removeprefix('evolution_')
        label = f'{run.name} ({job})'
        style = '-' if outcome['qualified_tracking_20M'] else '--'
        histories = list(run.glob('*.hst'))
        if histories:
            rows = numeric_rows(histories[0])
            axes[0].semilogy([r[0] for r in rows], [r[3] for r in rows],
                             linestyle=style, label=label)
        ff = list(run.glob('*.horizon_summary_0.txt'))
        if ff:
            rows = numeric_rows(ff[0])
            successful = outcome['successful_horizon_times']
            rows = [r for r in rows if any(abs(r[1] - t) < 5e-4 for t in successful)]
            times = [r[1] for r in rows]
            masses = [r[2] for r in rows]
            residuals = [math.sqrt(max(0, r[8])) * r[2] for r in rows]
        else:
            bha = list((run / 'horizon').glob('BHaHAHA_diagnostics.ah*.gp'))
            rows = numeric_rows(bha[0]) if bha else []
            times = [r[1] for r in rows]
            masses = [r[24] for r in rows]
            residuals = [r[14] for r in rows]
        axes[1].plot(times, [mass - 1 for mass in masses],
                     linestyle=style, label=label)
        axes[2].semilogy(times, residuals, linestyle=style, label=label)
        peaks[str(run.relative_to(root))] = slice_peaks(run)
        comparison = peak_horizon_comparison(run, peaks[str(run.relative_to(root))],
                                             outcome)
        if comparison is not None:
            peak_comparisons[str(run.relative_to(root))] = comparison
    axes[0].set_ylabel(r'$\int_{\chi\geq0.0625} H^2\,dV$')
    axes[1].set_ylabel('Horizon mass / rest mass − 1')
    axes[1].axhline(0, color='black', lw=0.7, ls='--')
    axes[2].set_ylabel('RMS expansion × mass')
    axes[2].set_xlabel('Time / rest mass')
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('Solid: qualified 20 M tracking; dashed: incomplete or failed tracking',
                 fontsize=10)
    fig.savefig(root / 'evolution_diagnostics.png', dpi=180)
    fig.savefig(root / 'evolution_diagnostics.pdf')
    plt.close(fig)
    (root / 'constraint_slice_peaks.json').write_text(
        json.dumps(peaks, indent=2, allow_nan=False) + '\n')
    (root / 'constraint_peak_horizon_comparison.json').write_text(
        json.dumps(peak_comparisons, indent=2, allow_nan=False) + '\n')
    selected = [run for run in runs if run.name in
                ('g1_fastflow_L5', 'g5_fastflow_L6', 'g5_fastflow_L7')]
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3.5 * len(selected)),
                             squeeze=False, constrained_layout=True)
    for run, pair in zip(selected, axes):
        paths = sorted((run / 'tab').glob('*.con.*.tab*'))
        available = [(slice_data(path)[0], path) for path in paths]
        chosen = dict.fromkeys(min(available, key=lambda item: abs(item[0] - target))[1]
                               for target in (0, 1, 5, 10, 20))
        tracker = numeric_rows(next(run.glob('*.co_0.txt')))
        for path in chosen:
            time, rows = slice_data(path)
            _, zrows = slice_data(path.with_name(path.name.replace('.con.', '.z4c.')))
            chi = {(r['gid'], r['i']): r['z4c_chi'] for r in zrows}
            rows = sorted(rows, key=lambda r: r['x1v'])
            values = [abs(r['con_H']) if chi[(r['gid'], r['i'])] >= 0.0625
                      else math.nan for r in rows]
            center = min(tracker, key=lambda r: abs(r[1] - time))[2]
            pair[0].semilogy([r['x1v'] for r in rows],
                             values, label=f't={time:.2f}')
            pair[1].semilogy([r['x1v'] - center for r in rows],
                             values, label=f't={time:.2f}')
        pair[0].set_xlabel('x / rest mass')
        pair[1].set_xlabel('(x − tracker x) / rest mass')
        pair[1].set_xlim(-4, 4)
        for ax in pair:
            ax.set_ylabel('|H|, chi ≥ 0.0625')
            ax.set_title(run.name)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.savefig(root / 'constraint_profiles.png', dpi=180)
    fig.savefig(root / 'constraint_profiles.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
