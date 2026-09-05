"""Plot measured CUDA pulse convergence; retain underresolved coarse points."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('summary', type=Path)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
report = json.loads(args.summary.read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for family, marker in zip(['p', 'Q', 'L', 'B'], ['o', 's', '^', 'x']):
    rows = sorted([r for r in report['runs'] if r['family'] == family
                   and r['direction'] == 1 and r['rate'] == 1
                   and r['amplitude'] == 1e-8], key=lambda r: r['n'])
    h = np.array([8/r['n'] for r in rows])
    error = np.array([r['error_l2_over_amplitude'] for r in rows])
    axes[0].loglog(h, error, marker+'-', label=family)
    axes[1].loglog(h, [abs(r['speed_fit']+.5) for r in rows], marker+'-')
    axes[2].semilogx(h[1:], np.log(error[:-1]/error[1:])/np.log(h[:-1]/h[1:]), marker+'-')
axes[0].loglog(h, error[-1]*(h/h[-1])**2, 'k:', label='second order')
axes[0].set_ylabel('L2 pulse error / initial amplitude')
axes[0].legend()
axes[1].set_ylabel('Absolute error in transport speed')
axes[2].set_ylabel('Adjacent-resolution L2 order')
axes[2].axhline(2, color='k', ls=':')
for axis in axes:
    axis.set_xlabel('Grid spacing')
    axis.grid(alpha=.25, which='both')
fig.suptitle('CUDA compact reduction pulses, transverse derivatives, rate = 1, t = 1')
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
