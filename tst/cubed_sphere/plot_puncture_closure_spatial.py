#!/usr/bin/env python3
"""Plot spatial closure comparison, keeping incomplete temporal admission explicit."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument('review', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
j = json.loads(a.review.read_text())
checks = {x['region_field']: x for x in j['checks']}
fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
for col, field in enumerate(['H', 'M_conformal', 'Z_conformal']):
    for row, suffix in enumerate(['l2', 'linf']):
        x = checks['shell/fixed_outer_collar/' + field + '_' + suffix]
        ax = axes[row, col]
        for closure in ['quintic', 'interior']:
            ax.semilogy(j['resolutions'], x[closure], 'o-', label=closure)
        ax.set(title=field[0] + (' RMS' if row == 0 else ' maximum'),
               xlabel='Balanced bulk N', xticks=j['resolutions'])
        ax.grid(alpha=.2)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
phase = ('Timestep controls complete; spatial closure criteria fail. '
         if 'all_fixed_time_controls_small' in j else 'Spatial comparison only: timestep controls incomplete. ')
fig.suptitle('Puncture outer collar r>0.9, t=0.1, matched closure pairs at Rb=0.98\n'
             + phase +
             'Immutable Minkowski reference; no long-time acceptance.')
fig.savefig(a.output, dpi=170)
fig.savefig(a.output.with_suffix('.pdf'))
