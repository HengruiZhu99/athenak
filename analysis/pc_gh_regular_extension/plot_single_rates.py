"""Compare matched long single-puncture rates, including unmasked bounds."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from make_inputs import parse
from read_complete_history import read_history

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis/pc_gh_localization'))
from plot_qualification import regional_rms

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('runs', type=Path, nargs='+')
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
for run in args.runs:
    params = parse((run/'used_input.athinput').read_text())
    rate = float(params['pc_gh']['reduction_rate'])
    hst, _ = read_history(next(run.glob('*.pcgh.hst')))
    bounds = np.genfromtxt(next(run.glob('*.pcgh-boundedness.dat')), names=True)
    if not all(np.isfinite(bounds[name]).all() for name in bounds.dtype.names):
        raise ValueError(f'Nonfinite full-volume data: {run}')
    region = regional_rms(hst, 'r1')
    label=f'rate {rate:g}'
    for axis, family in zip(axes[0], ['GH', 'reduction', 'curl']):
        axis.semilogy(hst['time'], region[family], label=label)
        axis.set_title(f'{family}: exterior r > 1 RMS')
    for axis, key, title in zip(axes[1], ['max_rho', 'pcgh_curl_Q_max', 'min_eigenvalue'],
            ['Maximum rho, full volume', 'Maximum Q curl, full volume', 'Minimum metric eigenvalue']):
        axis.plot(bounds['time'], bounds[key], label=label)
        axis.set_title(title)
for axis in axes.flat:
    axis.grid(alpha=.25)
    axis.set_xlabel('t / M')
axes[0, 0].legend()
fig.suptitle('Isotropic puncture, large static-AMR domain, finest spacing M/8')
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
