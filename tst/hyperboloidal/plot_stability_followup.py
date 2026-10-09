"""Plot the saved transition follow-up receipt; no simulation is rerun.

Usage: python plot_stability_followup.py receipt.json output.png
Requires numpy and matplotlib. The figure contrasts initial spatial convergence
with growing finite-time constraints; it does not establish pulse stability.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    runs = {item['name']: item['data']['cases'][0]
            for item in receipt['native_runs']}
    corrected = receipt['audits']['corrected_bins']['data']['cases']
    wide = {}
    for case in corrected:
        if case['name'].startswith('wide-N') and case['name'].endswith('degree2'):
            rows = case['measurements']
            wide[rows[0]['n']] = np.mean([row['Hdot_rms'] for row in rows])
    # Broad values are the production-control probes retained in the original
    # implementation receipt. The corrected-bin N48 value agrees exactly.
    broad = receipt['plot_data']['broad_Hdot']
    colors = ['#9a4b16', '#17617d']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), constrained_layout=True)
    ax = axes[0, 0]
    n = np.array(sorted(wide))
    ax.loglog(n, [broad[str(v)] for v in n], 'o-', color=colors[0],
              label='Transition .2–.8')
    ax.loglog(n, [wide[v] for v in n], 'o-', color=colors[1],
              label='Transition .05–.95')
    ax.loglog(n, wide[36] * (36 / n) ** 4, '--', color='.45', label='N⁻⁴ guide')
    ax.set(xlabel='Cells per Cartesian axis N', ylabel='RMS initial Hdot',
           title='Initial spatial defect: a=.5, symmetric quadratic ghosts')
    ax.set_xticks(n, labels=[str(v) for v in n])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=9)
    for ax, column, label in [(axes[0, 1], 2, 'Hamiltonian H'),
                              (axes[1, 0], 3, 'Momentum M'),
                              (axes[1, 1], 4, 'Spatial Z')]:
        for index, name in enumerate(['clean-kappa10-long',
                                      'clean-wide-kappa10-long']):
            rows = np.asarray(runs[name]['history']['rows'])
            selected = rows[:, 0] > 0
            ax.semilogy(rows[selected, 0], rows[selected, column],
                        color=colors[index], label=['.2–.8', '.05–.95'][index])
        ax.set(xlabel='Coordinate time (S=1)', ylabel=f'Unweighted RMS {label}',
               title=f'{label}: N24, κ=10, same actual timestep')
        ax.legend(title='Transition', frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.grid(True, which='major', alpha=.2)
    fig.suptitle('Initial convergence improves; long-run constraints still grow',
                 fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
