#!/usr/bin/env python3
"""Plot unprofiled paired windows, including variability and output costs."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('summary', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    d = json.loads(a.summary.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    modes = [m for m in ['legacy', 'masked', 'agreement'] if m in d['modes']]
    colors = {'legacy': '#666666', 'masked': '#377f9e', 'agreement': '#b96a35'}
    baseline = d['modes']['legacy']['mean_seconds_per_M0']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    metrics = []
    for i, mode in enumerate(modes):
        windows = [r for r in d['windows'] if r['mode'] == mode]
        ratios = [r['end_to_end_seconds_per_M0']/baseline for r in windows]
        ordinary = [r['interval_breakdown']['ordinary_median_seconds'] for r in windows]
        startup = [r['startup_to_cycle0_seconds'] for r in windows]
        final = [r['interval_breakdown']['final_step_plus_output_seconds'][0] for r in windows]
        for ax, values in zip(axes, [ratios, ordinary, startup]):
            ax.bar(i, statistics.mean(values), color=colors[mode], alpha=.75)
            ax.scatter(i+np.linspace(-.07, .07, len(values)), values, color='#222222', s=16, zorder=3)
        metrics.append({'mode': mode, 'windows': len(windows),
                        'wall_per_M0_mean': statistics.mean(r['end_to_end_seconds_per_M0'] for r in windows),
                        'wall_per_M0_stdev': statistics.stdev(r['end_to_end_seconds_per_M0'] for r in windows) if len(windows)>1 else None,
                        'median_step_mean': statistics.mean(r['typical_step_seconds'] for r in windows),
                        'mean_step_including_diagnostics_final_output': statistics.mean(r['average_step_seconds'] for r in windows),
                        'ordinary_step_median_mean': statistics.mean(ordinary),
                        'startup_mean_seconds': statistics.mean(startup),
                        'final_step_plus_output_mean_seconds': statistics.mean(final),
                        'mean_wall_ratio': statistics.mean(ratios)})
    axes[0].axhline(1.15, color='#a12d28', ls='--', lw=1.4, label='1.15 limit')
    axes[0].set(ylabel='Relative to mean baseline wall/M0', title='End-to-end cost')
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set(ylabel='Seconds per ordinary step', title='Median step cost')
    axes[2].set(ylabel='Seconds to native cycle 0', title='Startup, including initial output')
    for ax in axes:
        ax.set(xticks=range(len(modes)), xticklabels=modes)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    steps = ', '.join(str(n) for n in sorted({r['steps'] for r in d['windows']}))
    fig.suptitle('Two-node N16 comparison: '+steps+'-step windows'+(' — partial protocol' if not d['complete_protocol'] else ''), fontsize=13)
    fig.savefig(a.output/'paired-performance.png', dpi=180)
    lines = ['| Mode | Windows | Mean window-median step (s) | Mean interval including diagnostics/final output (s) | Wall/M0 mean ± sample SD (s) | Mean ratio |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in metrics:
        spread = '%.3f (single window)' % r['wall_per_M0_mean'] if r['wall_per_M0_stdev'] is None else '%.3f ± %.3f' % (r['wall_per_M0_mean'], r['wall_per_M0_stdev'])
        lines.append('| %s | %d | %.6f | %.6f | %s | %.4f |' %
                     (r['mode'], r['windows'], r['median_step_mean'],
                      r['mean_step_including_diagnostics_final_output'], spread, r['mean_wall_ratio']))
    lines += ['', 'End-to-end includes launch, initialization, evolution, and all outputs. Dots show individual windows; they are not confidence intervals.',
              '', 'The final-step-plus-output interval includes one PDE step and is not an instrumented pure I/O timer. Scientific benefit is assessed separately.', '']
    (a.output/'performance-table.md').write_text('\n'.join(lines))
    receipt = {'metrics': metrics, 'protocol_complete': d['complete_protocol'],
               'summary_sha256': hashlib.sha256(a.summary.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'performance-plot-data.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
