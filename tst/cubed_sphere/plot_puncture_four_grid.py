#!/usr/bin/env python3
"""Retain all four puncture grids when displaying the later three-grid review."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('first_review', type=Path)
p.add_argument('last_review', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
first = {x['region_field']: x for x in json.loads(a.first_review.read_text())['checks']}
last = {x['region_field']: x for x in json.loads(a.last_review.read_text())['checks']}
regions = {'bulk/fixed_interior': 'Bulk interior', 'bulk/fixed_interface': 'Bulk interface',
           'shell/fixed_inner_shell': 'Inner shell', 'shell/fixed_transition': 'Shell transition',
           'shell/fixed_outer_collar': 'Outer collar'}
fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
rows = []
for col, keys in enumerate([['H_l2', 'H_linf'], ['M_conformal_l2', 'M_conformal_linf'],
                            ['Z_conformal_l2', 'Z_conformal_linf']]):
    for row, key in enumerate(keys):
        ax = axes[row, col]
        for region, label in regions.items():
            k = region + '/' + key
            assert first[k]['values'][1:] == last[k]['values'][:2]
            values = first[k]['values'] + [last[k]['values'][-1]]
            rows.append(dict(region_field=k, resolutions=[16, 24, 32, 40], values=values))
            ax.semilogy([16, 24, 32, 40], values, 'o-', label=label)
        ax.set(title=key.replace('_conformal', '').replace('_l2', ' RMS').replace('_linf', ' maximum'),
               xlabel='Bulk N; balanced angular/radial refinement', xticks=[16, 24, 32, 40])
        ax.grid(alpha=.2)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
fig.suptitle('Native puncture t=0.1, Rb=0.98, immutable Minkowski reference\n'
             'All four grids retained; N24–32–40 outer-collar H/M nonmonotonic, full convergence fails')
fig.savefig(a.output, dpi=170)
fig.savefig(a.output.with_suffix('.pdf'))
a.output.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')
