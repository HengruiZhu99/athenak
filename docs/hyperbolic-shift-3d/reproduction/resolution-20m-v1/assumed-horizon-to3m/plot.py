#!/usr/bin/env python3
"""Plot diagnostic-only checkpoint reanalysis, with provisional horizon cuts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REP = HERE.parents[1]
OLD = REP / 'n16-timestep-to3m-v1/n16-original-dt-reproduction-v1/output'
values = {n: json.loads((HERE / f'n{n}-final.json').read_text()) for n in [16,24]}
parity = json.loads((HERE / 'n16-parity.json').read_text())
reference = json.loads((OLD / 'puncture-initial.report.json').read_text())
checks = {}
for field in ['H_L2', 'M_L2', 'Mr_L2', 'Theta_L2', 'Zr_L2']:
    native = field.replace('_L2', '_exterior_L2')
    if native in reference:
        checks['parity_' + field] = abs(parity['norms']['exterior'][field] - reference[native])
        assert np.isclose(parity['norms']['exterior'][field], reference[native], rtol=1e-10, atol=1e-13)
for n, report in [(16, OLD/'puncture.report.json'), (24, HERE.parent/'n24-at3m/puncture.report.json')]:
    native = json.loads(report.read_text())
    assert values[n]['t_over_M'] == 3 and values[n]['steps'] == native['steps']
    assert values[n]['cutoff'] == native['horizon_trial_radius']
    for field in ['H_L2', 'M_L2', 'Mr_L2', 'Zr_L2']:
        checks[f'n{n}_total_' + field] = abs(values[n]['norms']['total'][field]-native[field])
        assert np.isclose(values[n]['norms']['total'][field], native[field], rtol=1e-10, atol=1e-13)

plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('Constraint errors outside the assumed horizons at 3M', fontsize=17, y=.98)
fig.text(.5,.93,'Checkpoint reanalysis using each run’s recorded candidate radius', ha='center', fontsize=11)
summary = {}
for ax, field, title in zip(axes.flat, ['H_L2','Mr_L2','M_L2','Theta_L2'],
                            ['Hamiltonian', 'Radial momentum', 'Invariant momentum', r'Z4 scalar $\Theta$']):
    v = [values[n]['norms']['exterior'][field] for n in [16,24]]
    reduction = 1-v[1]/v[0]
    summary[field] = dict(N16=v[0],N24=v[1],reduction_fraction=reduction)
    bars = ax.bar(['N16','N24'], v, color=['#b45418','#1769aa'], width=.55)
    ax.bar_label(bars, labels=[f'{x:.5g}' for x in v], padding=5, fontsize=11)
    ax.set_ylim(0, max(v)*1.25)
    direction = 'lower' if reduction >= 0 else 'higher'
    ax.set_title(f'{title}: {100*abs(reduction):.1f}% {direction} on N24', fontsize=12)
    ax.set_ylabel('Exterior RMS norm')
    ax.grid(axis='y', alpha=.22); ax.set_axisbelow(True)
fig.subplots_adjust(top=.84,bottom=.17,hspace=.44,wspace=.34)
fig.text(.5,.055,'Cuts: r > 0.07519062 (N16), r > 0.07502426 (N24).\n'
         'Normalized compactified-coordinate volume; candidate horizons assumed valid.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(HERE/f'exterior-constraints-at3m.{ext}',dpi=180)
v = [values[n]['norms']['exterior']['Zr_L2'] for n in [16,24]]
summary['Zr_L2'] = dict(N16=v[0],N24=v[1],reduction_fraction=1-v[1]/v[0])
(HERE/'comparison.json').write_text(json.dumps(dict(exterior=summary,parity_absolute_differences=checks),indent=2)+'\n')
print(json.dumps(summary,indent=2))
