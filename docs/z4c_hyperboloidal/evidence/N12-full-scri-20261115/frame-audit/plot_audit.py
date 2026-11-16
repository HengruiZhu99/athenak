"""Render the saved read-only audit; no metric cache is needed."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
report = json.loads((root/'audit.json').read_text())
assert hashlib.sha256((root/'scri_frame_audit.py').read_bytes()).hexdigest() == report['audit_sha256']
rows = report['samples']
keys = list(rows[0])
with (root/'decomposition.csv').open('w') as stream:
    writer = csv.DictWriter(stream, keys, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
column = lambda key: np.array([r[key] for r in rows])
t = column('native_time_M0')
raw = column('ricci_N20_real')
correction = column('hessian_N20_real') + column('quadratic_N20_real')
total = column('formal_N20_real')
np.testing.assert_allclose(raw + correction, total, atol=2e-13, rtol=2e-12)

fig, axes = plt.subplots(2, 1, figsize=(9, 6.7), sharex=True,
                         gridspec_kw={'height_ratios': [1.6, 1]}, layout='constrained')
axes[0].plot(t, total, color='#222222', lw=2, label='Formal total')
axes[0].plot(t, raw, color='#1673aa', lw=1.6, label='Ricci contribution in same final dyad')
axes[0].plot(t, correction, color='#c55d15', lw=1.6, label='Conformal Hessian + gradient corrections')
axes[0].axhline(0, color='0.7', lw=.7)
axes[0].set_ylabel(r'Re $N_{20}$ contributions')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title('N12 at scri: frame corrections dominate much of the late diagnostic', fontsize=12)
axes[1].plot(t, column('round_scale_mean'), color='#8b4296', lw=1.8,
             label='Mean cut scale defect (target 0)')
axes[1].plot(t, column('conformal_hessian_tf_max'), color='#388451', lw=1.6,
             label='Max component of trace-free Hessian of Ω (target 0)')
axes[1].set_ylabel('Geometry diagnostics')
axes[1].set_xlabel(r'Native simulation time $t/M_0$ (same cuts for all terms)')
axes[1].legend(loc='upper left', fontsize=9)
for ax in axes:
    ax.grid(alpha=.18)
    ax.set_xlim(0, 60)
fig.suptitle('Unvalidated radiation diagnostic — not a physical Bondi waveform',
             fontsize=11, color='#9c2929')
fig.savefig(root/'decomposition.png', dpi=180)
fig.savefig(root/'decomposition.pdf')
plt.close(fig)
print(f'Verified audit script hash and additive identity for {len(rows)} cuts.')
