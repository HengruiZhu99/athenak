"""Effective H residual order from verified, time-matched own-candidate histories."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'n16-n20'
comparison = json.loads((OUT / 'comparison.json').read_text())
data, provenance = {}, {}
for n, key, raw in [(16, 'N16*', 'restored-six-shells'),
                    (20, 'N20', 'restored-six-shells-resolution')]:
    segment, = comparison['cases'][key]['segments']
    name, record = next((k, v) for k, v in segment['artifacts'].items()
                        if k.endswith('/puncture.history.csv'))
    path = ROOT / 'output' / raw / 'aurora' / name
    content = path.read_bytes()
    assert len(content) == record['bytes']
    assert hashlib.sha256(content).hexdigest() == record['sha256']
    data[n] = np.genfromtxt(path, delimiter=',', names=True)
    provenance[str(n)] = dict(path=str(path.relative_to(ROOT)), **record)

a, b = data[16], data[20]
assert len(a) == len(b)
assert np.all(np.diff(a['t_over_M']) > 0)
assert np.all(np.diff(b['t_over_M']) > 0)
time_delta = float(np.max(np.abs(a['t_over_M'] - b['t_over_M'])))
assert time_delta < 1e-9, 'Time matching required; no implicit interpolation'
t = (a['t_over_M'] + b['t_over_M']) / 2
h16, h20 = a['H_exterior_L2'], b['H_exterior_L2']
assert np.all(np.isfinite(h16) & (h16 > 0) & np.isfinite(h20) & (h20 > 0))
p = np.log(h16 / h20) / np.log(20 / 16)
endpoint = json.loads((OUT / 'orders.json').read_text())['endpoint']['H_L2']['effective_order']
rows = np.column_stack([t, h16, h20, p, a['exterior_cutoff'], b['exterior_cutoff'],
                       a['horizon_found'], b['horizon_found']])
np.savetxt(OUT / 'hamiltonian-order-history.csv', rows, delimiter=',',
           header='t_over_M,H16_own_candidate,H20_own_candidate,p_effective,cutoff16,cutoff20,horizon_accepted16,horizon_accepted20', comments='')
summary = dict(formula='ln(H16_exterior/H20_exterior)/ln(20/16)',
               history_artifacts=provenance, matched_rows=len(t), max_time_mismatch=time_delta,
               interpolation='none', smoothing='none',
               p_initial=float(p[0]), p_final_own_candidate=float(p[-1]),
               p_final_common_cutoff_012=endpoint, p_range=[float(p.min()), float(p.max())],
               max_cutoff_difference=float(np.max(np.abs(a['exterior_cutoff']-b['exterior_cutoff']))),
               qualifications=['Own-candidate exterior regions differ between resolutions and change with time.',
                               'Candidate regions remain recorded when the horizon finder is not accepted; flags are preserved in CSV.',
                               'No common fixed-cutoff historical series is available; only the 10M diamond has common cutoff r>0.12.',
                               'Two-resolution residual estimate with joint space/time refinement and matched clipped boundary penalty.',
                               'N16 native completion retains its historical penalty-policy failure.'])
(OUT / 'hamiltonian-order-history.json').write_text(json.dumps(summary, indent=2)+'\n')
fig, (ax, cut) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
ax.plot(t, p, '.-', color='#2166ac', lw=1.5, ms=3, label='N16 / N20: each run’s candidate exterior')
for order, style in [(2, ':'), (3, '-.'), (4, '--')]:
    ax.axhline(order, color='gray', ls=style, lw=1, label=f'p = {order}')
ax.plot(10, endpoint, 'D', color='#d95f02', ms=7, label=f'10M, common r > 0.12: p = {endpoint:.2f}')
ax.set(ylabel=r'Effective order $p_H(t)$', title='Exterior Hamiltonian constraint: N16–N20 effective order')
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=.2)
cut.plot(t, a['exterior_cutoff'], color='#238b45', label='N16 cutoff')
cut.plot(t, b['exterior_cutoff'], '--', color='#2166ac', label='N20 cutoff')
cut.set(xlabel=r'$t/M$', ylabel='Exterior cutoff r', xlim=(0, 10.15))
cut.legend(fontsize=9, loc='upper left'); cut.grid(alpha=.2)
fig.tight_layout(rect=(0,.09,1,1))
fig.subplots_adjust(left=.12)
fig.text(.5,.025,'Recorded times matched to < 2×10⁻¹² M; no interpolation or smoothing.\nMoving candidate cutoffs: this curve is not a fixed-domain convergence proof.', ha='center', fontsize=9)
fig.savefig(OUT / 'hamiltonian-order-history.png', dpi=180)
plt.close(fig)
print(json.dumps(summary, indent=2))
