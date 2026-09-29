#!/usr/bin/env python3
"""Plot saved campaign data and locate Hamiltonian peaks on the x-axis slice."""
import argparse
import json
import math
import re
from pathlib import Path

from analyze import numeric_rows


def slice_data(path):
    lines = path.read_text().splitlines()
    time = float(re.search(r'time=([\d.eE+-]+)', lines[0])[1])
    names = lines[1].split()[1:]
    return time, [dict(zip(names, row)) for row in numeric_rows(path)]


def slice_peaks(run):
    trackers = list(run.glob('*.co_0.txt'))
    tracker = numeric_rows(trackers[0]) if trackers else []
    peaks = []
    for path in sorted((run / 'tab').glob('*.con.*.tab')):
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
    for run in runs:
        label = run.name
        histories = list(run.glob('*.hst'))
        if histories:
            rows = numeric_rows(histories[0])
            axes[0].semilogy([r[0] for r in rows], [r[3] for r in rows], label=label)
        ff = list(run.glob('*.horizon_summary_0.txt'))
        if ff:
            rows = numeric_rows(ff[0])
            times = [r[1] for r in rows]
            masses = [r[2] for r in rows]
            residuals = [math.sqrt(max(0, r[8])) * r[2] for r in rows]
        else:
            bha = list((run / 'horizon').glob('BHaHAHA_diagnostics.ah*.gp'))
            rows = numeric_rows(bha[0]) if bha else []
            times = [r[1] for r in rows]
            masses = [r[24] for r in rows]
            residuals = [r[14] for r in rows]
        axes[1].plot(times, masses, label=label)
        axes[2].semilogy(times, residuals, label=label)
        peaks[str(run.relative_to(root))] = slice_peaks(run)
    axes[0].set_ylabel(r'$\int_{\chi\geq0.0625} H^2\,dV$')
    axes[1].set_ylabel('Horizon mass / rest mass')
    axes[1].axhline(1, color='black', lw=0.7, ls='--')
    axes[2].set_ylabel('RMS expansion × mass')
    axes[2].set_xlabel('Time / rest mass')
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(root / 'evolution_diagnostics.png', dpi=180)
    fig.savefig(root / 'evolution_diagnostics.pdf')
    plt.close(fig)
    (root / 'constraint_slice_peaks.json').write_text(
        json.dumps(peaks, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
