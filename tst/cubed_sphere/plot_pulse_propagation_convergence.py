#!/usr/bin/env python3
"""Native GPU pulse norms at common times; preserve failed growth-gate context."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument('review', type=Path)
p.add_argument('initial', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
j = json.loads(a.review.read_text())
assert j['common_target_reached']
new = {x['region_field']:x for x in j['spatial_checks']}
old = {x['region_field']:x for x in json.loads(a.initial.read_text())['checks']}
fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
for row, region in enumerate(['shell/all','shell/fixed_outer_collar']):
    for col, field in enumerate(['H_l2','M_conformal_l2','Z_conformal_l2']):
        key = region+'/'+field
        ax = axes[row, col]
        ax.semilogy([16,24,32], old[key]['values'], 'o--', label='t=0.1')
        ax.semilogy([16,24,32], new[key]['values'], 'o-', label='t=0.2')
        ax.set(title=('Whole shell ' if row == 0 else 'Outer collar ') + field[0] + ' RMS',
               xticks=[16,24,32], xlabel='Balanced bulk N')
        ax.grid(alpha=.2)
        ax.text(.04,.06,'Finest order at t=.2: %.2f' % new[key]['orders'][-1],
                transform=ax.transAxes, fontsize=10)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
fig.suptitle('Native 20% quadrupolar pulse, quintic Rb=0.98, immutable Minkowski reference\n'
             'Short-time refinement only: original growth monitors still fail; long stability unproven.')
fig.savefig(a.output,dpi=170)
fig.savefig(a.output.with_suffix('.pdf'))
