#!/usr/bin/env python3
"""Plot matched-time collar sensitivities without suggesting long stability."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument('review', type=Path)
p.add_argument('closure', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
r = {x['region_field']: x for x in json.loads(a.review.read_text())['checks']}
c = {x['region_field']: x for x in json.loads(a.closure.read_text())['checks']}
fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
for col, field in enumerate(['H', 'M_conformal', 'Z_conformal']):
    for row, suffix in enumerate(['l2', 'linf']):
        key = 'shell/fixed_outer_collar/' + field + '_' + suffix
        x, z = r[key], c[key]
        assert x['baseline'] == z['quintic']
        baseline = x['baseline']
        ratios = [1., x['radial_only']/baseline, x['angular_only']/baseline,
                  z['interior']/baseline]
        ax = axes[row, col]
        ax.bar(['Baseline', 'Radial', 'Angular', 'Interior'], ratios,
               color=['gray', 'tab:blue', 'tab:orange', 'tab:purple'])
        ax.axhline(1., color='black', linewidth=.7)
        ax.set(title=field[0] + (' RMS' if row == 0 else ' maximum'),
               ylabel='Ratio to quintic B32/A24/R164', ylim=(0, max(ratios)*1.2))
        ax.grid(axis='y', alpha=.2)
fig.suptitle('Puncture outer collar r>0.9 at t=0.1, Rb=0.98\n'
             'Radial R205; angular A30; interior closure on original grid. '
             'Immutable Minkowski reference; no horizon exclusion.')
fig.savefig(a.output, dpi=170)
fig.savefig(a.output.with_suffix('.pdf'))
