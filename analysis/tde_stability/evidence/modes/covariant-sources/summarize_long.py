"""Summarize the validated sigma=1 restart continuation without equating it to stability."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = [ROOT / 'covariant_const10', ROOT / 'covariant_const10_continuation']


def merged(name):
    arrays = [np.atleast_2d(np.loadtxt(d / name)) for d in RUNS]
    data = np.concatenate(arrays)
    data = data[np.argsort(data[:, 0], kind='stable')]
    _, index = np.unique(data[:, 0], return_index=True)
    return data[index]


def growth(t, y, lo, hi):
    good = (t >= lo) & (t <= hi) & (y > 0) & np.isfinite(y)
    x, v = t[good], np.log(y[good])
    p = np.polyfit(x, v, 1)
    err = v - np.polyval(p, x)
    return {'window_M': [lo, hi], 'actual_window_M': [float(x[0]), float(x[-1])],
            'gamma_per_M': float(p[0]),
            'log_fit_R2': float(1 - np.sum(err**2) / np.sum((v-v.mean())**2)),
            'last_to_first_ratio': float(np.exp(v[-1]-v[0]))}


u = merged('ks_background.user.hst')
z = merged('ks_background.z4c.user.hst')
ext = np.sqrt(np.maximum(z[:, 9], 0))
interior = np.sqrt(np.maximum(z[:, 14], 0))
windows = [(100, 200), (200, 300), (300, 500), (500, 750), (750, 1000)]
validity = json.loads((RUNS[-1] / 'checkpoint-validity.json').read_text())
log = (RUNS[-1] / 'run.log').read_text()
result = {
    'configuration': 'G2, coupled covariant source prototype, constant sigma=1, dx=0.25M, dt=0.0375M',
    'time_M': float(u[-1, 0]), 'target_M': 1000,
    'termination': 'simulation time target' if 'Terminating on time limit' in log and u[-1,0] == 1000 else 'inspect log',
    'exit_code': int((RUNS[-1] / 'exit_code.txt').read_text()),
    'restart_at_M': 300, 'history_finite': bool(np.isfinite(u).all() and np.isfinite(z).all()),
    'invalid_or_recovery_log': any(s in log for s in ['C2P_INVALID_ADM_INPUT', 'Z4C_INVALID_STATE', 'FATAL ERROR']),
    'final_checkpoint': {k: v for k, v in validity.items() if k != 'files'},
    'final': {'Theta_max': float(u[-1,9]), 'Theta_exterior_L2': float(ext[-1]),
              'Theta_interior_L2': float(interior[-1]), 'alpha_res_max': float(u[-1,12]),
              'Gamma_res_max': float(u[-1,15]), 'passive_fluid_rho_max': float(u[-1,2]),
              'badmetrics': float(u[-1,11])},
    'windows': {'Theta_max': [growth(u[:,0],u[:,9],a,b) for a,b in windows],
                'Theta_exterior_L2': [growth(z[:,0],ext,a,b) for a,b in windows]},
    'interpretation': 'Valid finite target completion, not stability. Fits vary with the window and are not one identified eigenmode. A separately verified oscillatory growing mode has gamma about 0.0018/M. Matter feedback is disabled; passive fluid behavior is not atmosphere validation.'}
(ROOT / 'long-results.json').write_text(json.dumps(result, indent=2) + '\n')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained', facecolor='white')
for a, col, label in [(ax[0],9,r'Active max $|\Theta|$'), (ax[1],None,r'$\|\Theta\|_{L^2(r>1M)}$')]:
    for folder, name, color in [('baseline',r'Original Z4c, $\sigma=0.1\alpha$','#386cb0'),
                                ('scaled_03',r'Original Z4c, $\sigma=0.3$','#31985f')]:
        d = ROOT.parent / 'lapse-damping' / folder
        h = np.atleast_2d(np.loadtxt(d / ('ks_background.user.hst' if col else 'ks_background.z4c.user.hst')))
        a.semilogy(h[:,0], h[:,col] if col else np.sqrt(np.maximum(h[:,9],0)), color=color, label=name)
    d = ROOT / 'covariant_const03'
    h = np.atleast_2d(np.loadtxt(d / ('ks_background.user.hst' if col else 'ks_background.z4c.user.hst')))
    a.semilogy(h[:,0], h[:,col] if col else np.sqrt(np.maximum(h[:,9],0)), color='#e68a24', label=r'Coupled sources, $\sigma=0.3$')
    t, y = (u[:,0],u[:,col]) if col else (z[:,0],ext)
    a.semilogy(t, np.where(y>0,y,np.nan), color='#a34e9c', label=r'Coupled sources, $\sigma=1$')
    a.axvline(300, color='0.6', lw=.8, ls=':')
    a.set(xlabel=r'Time / $M$', ylabel=label, xlim=(0,1000))
    a.grid(alpha=.2)
ax[0].legend(fontsize=7, loc='lower right')
fig.savefig(ROOT / 'long-comparison.png', dpi=180)
plt.close(fig)
print(json.dumps(result, indent=2))
