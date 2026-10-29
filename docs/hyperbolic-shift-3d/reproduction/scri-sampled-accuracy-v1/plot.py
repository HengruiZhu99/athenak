"""Plot retained, absolute manufactured errors; no native evaluation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/'summary.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
for measure in ('normal', 'full_rate'):
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), sharex=True)
    methods = [('discrete_AD', 'Existing AD', '#1765a0'),
               ('sampled_native_numerators', 'Sampled native N', '#cf671c'),
               ('sampled_exact_numerators', 'Sampled analytic N', '#328653')]
    if measure == 'full_rate':
        methods = [('full_rate_discrete_AD', 'Existing AD', '#1765a0'),
                   ('full_rate_sampled_native', 'Sampled native N', '#cf671c')]
    for axis, fixture in zip(axes.flat, data['fixtures']):
        for name, label, color in methods:
            rows = fixture['methods'][name]
            axis.loglog([r['n'] for r in rows], [r['l2'] for r in rows], 'o-', color=color,
                        label=f"{label}; last order {rows[-1]['observed_order_l2']:.2f}")
        axis.set_title(fixture['fixture'].replace('_', ' ') +
                       (' — analytic N=C=0' if fixture['exact_boundary_regular'] else ' — off regularity'))
        axis.set_xticks([8, 12, 16, 24], labels=['8', '12', '16', '24'])
        axis.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
        axis.grid(alpha=.2, which='major')
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Cells per block axis n')
    title = 'Normal pole derivative' if measure == 'normal' else 'Complete free-geometry surface rate'
    figure.suptitle(title+' — manufactured consistency study', fontsize=14)
    figure.supylabel('Absolute L2 error (reference area, Frobenius)', x=.02)
    figure.text(.5, .015, 'Four resolutions; observed orders are not imposed. No evolution or stability test.\n'
                'p1 full-rate target includes its unchanged h-dependent penalties; p2 has zero analytic pole and regularity traces.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(.04, .07, 1, .95), h_pad=2)
    figure.savefig(HERE/f'{measure}-accuracy.png', dpi=160)
    figure.savefig(HERE/f'{measure}-accuracy.pdf')
    plt.close(figure)
