#!/usr/bin/env python3
"""Plot manufactured spatial errors, separating ghost transfer from mapped FD."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
assert json.loads((a.raw/'completion.json').read_text())['cases'] == 12
schemes = ['matched-nonoverlap', 'control-l3', 'thin-l1']
labels = ['Cube-to-sphere, no overlap', 'Spherical: core L3, shell L3', 'Spherical: core L1, shell L3']
resolutions = np.array([12, 16, 24]); data = {}
for scheme in ['legacy-v7', *schemes]:
    for n in resolutions:
        record = json.loads((a.raw/f'{scheme}-n{n}.json').read_text())
        assert record['scheme'] == scheme and record['n'] == n and not record['vertex_centered']
        data[scheme, int(n)] = record
rows = []
for (scheme, n), record in data.items():
    for region, modes in record['errors'].items():
        for mode, values in modes.items():
            rows.append(dict(scheme=scheme, n=n, region=region, mode=mode, **values))
with (a.out/'errors.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
summary = dict(scope='Manufactured scalar spatial derivatives only. Common physical masks but mesh-specific midpoint quadratures and nodes. Does not measure paired-penalty dynamics, reflected waves, puncture fidelity or global stability.', n16={}, apparent_orders_16_to_24={})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for row, field in enumerate(['oblique', 'radial']):
    for col, region in enumerate(['core-outer', 'shell-common-edges']):
        key = field+'/'+region; ax = axes[row, col]
        summary['n16'][key] = {}
        summary['apparent_orders_16_to_24'][key] = {}
        for color, scheme, label in zip(['C0', 'C1', 'C2'], schemes, labels):
            for mode, style in [('coupled', '-'), ('analytic_ghosts', ':')]:
                y = np.array([data[scheme, int(n)]['errors'][key][mode]['hessian_rms'] for n in resolutions])
                assert np.isfinite(y).all() and np.all(y > 0)
                ax.loglog(resolutions, y, style, marker='o' if scheme != 'thin-l1' else 'x', color=color)
            values = data[scheme, 16]['errors'][key]
            summary['n16'][key][scheme] = {mode: v['hessian_rms'] for mode, v in values.items()}
            summary['apparent_orders_16_to_24'][key][scheme] = float(np.log(
                values['coupled']['hessian_rms']/data[scheme, 24]['errors'][key]['coupled']['hessian_rms'])/np.log(1.5))
        ax.set_xticks(resolutions, [str(n) for n in resolutions]); ax.set_xlabel('Shell intervals per block n')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel('Hessian tensor RMS error')
        ax.set_title(f'{field.capitalize()} field: '+('outer Cartesian region' if col == 0 else 'common shell edge region'))
        ax.grid(True, which='both', alpha=.2)
handles = [Line2D([], [], color=f'C{i}', label=label) for i, label in enumerate(labels)]
handles += [Line2D([], [], color='k', linestyle=style, label=label) for style, label in [('-', 'Interpolated ghosts'), (':', 'Exact analytic ghosts')]]
fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8)
fig.suptitle('Cell-centered transition audit — analytic truth, no time evolution')
fig.savefig(a.out/'transition-errors.png', dpi=180); fig.savefig(a.out/'transition-errors.pdf')
fig2, axes2 = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
summary['evolved_interface_n16'] = {}
for row, field in enumerate(['oblique', 'radial']):
    for col, region in enumerate(['core-interface-stencil', 'shell-interface-stencil']):
        key = field+'/'+region; ax = axes2[row, col]
        summary['evolved_interface_n16'][key] = {}
        for color, scheme in zip(['C0', 'C1', 'C2'], schemes):
            for mode, style in [('coupled', '-'), ('analytic_ghosts', ':')]:
                y = [data[scheme, int(n)]['errors'][key][mode]['hessian_rms'] for n in resolutions]
                ax.loglog(resolutions, y, style, marker='o' if scheme != 'thin-l1' else 'x', color=color)
            summary['evolved_interface_n16'][key][scheme] = {
                mode: v['hessian_rms'] for mode, v in data[scheme, 16]['errors'][key].items()}
        ax.set_xticks(resolutions, [str(n) for n in resolutions]); ax.set_xlabel('Shell intervals per block n')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel('Hessian tensor RMS error')
        ax.set_title(f'{field.capitalize()} field: '+('outer evolved core rows' if col == 0 else 'outer evolved shell angular rows'))
        ax.grid(True, which='both', alpha=.2)
fig2.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8)
fig2.suptitle('Actual artificial-interface stencils — includes extra evolved copies\nTwo rows per interface; physical locations differ across meshes')
fig2.savefig(a.out/'evolved-interface-errors.png', dpi=180); fig2.savefig(a.out/'evolved-interface-errors.pdf')
(a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
