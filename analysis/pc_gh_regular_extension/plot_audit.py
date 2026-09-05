"""Plot measured kernel audit residuals; these are not evolution results."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('run', type=Path)
args = ap.parse_args()
p = np.genfromtxt(args.run/'principal-errors.csv', delimiter=',', names=True)
f = np.genfromtxt(args.run/'fourier-spectrum.csv', delimiter=',', names=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3), layout='constrained')
ax[0].bar(p['case'], p['max_error']/np.finfo(float).eps, color='#206b83', width=.65)
ax[0].set(xlabel='Independent background / direction case',
          ylabel='Largest principal-matrix residual / machine epsilon',
          title='Compiled RHS versus independent symbol', xticks=p['case'])
ax[0].set_ylim(0, 2.6)
ax[1].plot(f['k'], f['max_real_part'], 'o-', color='#206b83')
ax[1].axhline(0, color='#65737b', linewidth=.8)
ax[1].set_xscale('symlog', linthresh=.01)
ax[1].set_xlim(0, 150)
ax[1].set(xlabel='Fourier wave number k', ylabel='Largest measured real eigenvalue',
          title='Full 50-field production linearization')
ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax[1].text(.03, .87, 'Exact polynomial: Re(s) ≤ 0 for all real k\nSmall positive values are numerical roundoff',
           transform=ax[1].transAxes, fontsize=9)
fig.suptitle('Regular PC-GH extension — zero-step kernel audits, no CUDA evolution', fontsize=12)
for ext in ['png', 'pdf']:
    fig.savefig(args.run/f'kernel-audit.{ext}', dpi=170)
