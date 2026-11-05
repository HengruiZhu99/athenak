"""Admitted N12 history and exact-time early N16 residual comparison."""
from pathlib import Path
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
O = D / 'radial18-n12-segment01-analysis'
O.mkdir(exist_ok=True)
V = json.loads((D / 'radial18-n12-segment01/local-verification.json').read_text())
assert V['all_worker_artifacts_verified'] and V['admitted']
paths = {12: D / 'radial18-n12-segment01/output/state.history.jsonl',
         16: D / 'radial-control-segment01/output/state.history.jsonl'}
rows = {n: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        for n, p in paths.items()}

def fixed(row):
    q = [v for v in row['fixed'] if v['cut'] == .3]
    assert len(q) == 1
    return q[0]

pairs = []
for a in rows[12]:
    b = min(rows[16], key=lambda b: abs(a['time'] - b['time']))
    if abs(a['time'] - b['time']) > 1e-12:
        continue
    v = dict(time_M=3*a['time'], time_mismatch_M=3*abs(a['time']-b['time']))
    for k in ['H', 'M', 'Mr']:
        v[k] = dict(N12=fixed(a)[k], N16=fixed(b)[k],
                    order=math.log(fixed(a)[k]/fixed(b)[k])/math.log(16/12))
    pairs.append(v)
assert pairs
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
colors = {12: 'tab:blue', 16: 'tab:orange'}
for n, values in rows.items():
    for k, ls in [('H', '-'), ('M', '--')]:
        ax[0, 0].semilogy([3*r['time'] for r in values],
                          [fixed(r)[k] for r in values], ls,
                          color=colors[n], label=f'N{n} {k}')
    ax[1, 0].semilogy([3*r['time'] for r in values],
                      [r['horizon_trial_expansion_rms'] for r in values],
                      '.-', color=colors[n], label=f'N{n}')
    accepted = [r for r in values if r['horizon_found'] and r['mass'] is not None]
    ax[1, 1].plot([3*r['time'] for r in accepted],
                  [100*(3*r['mass']-1) for r in accepted], '.',
                  color=colors[n], label=f'N{n} accepted only')
for k in ['H', 'M', 'Mr']:
    ax[0, 1].plot([p['time_M'] for p in pairs], [p[k]['order'] for p in pairs],
                  '.-', label=k)
ax[0, 1].axhline(4, color='gray', ls=':')
ax[1, 0].axhline(.001, color='gray', ls=':', label='online RMS threshold')
labels = [('Fixed r > 0.3 constraints', 'Residual norm'),
          ('Exact common times only', 'Two-point residual order'),
          ('Complete horizon candidates', 'Expansion RMS'),
          ('No mass inferred for rejected candidates', 'Rest-mass error (%)')]
for axis, (title, ylabel) in zip(ax.flat, labels):
    axis.set(title=title, xlabel='t/M', ylabel=ylabel)
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
fig.suptitle('Stationary radial18: N12 walltime segment vs admitted N16 short control')
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(O / ('history-and-early-order.' + ext), dpi=170)
last = rows[12][-1]
result = dict(
    input_sha256={str(n): hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()},
    last_M=3*last['time'], fixed03=fixed(last),
    horizon_rows=len(rows[12]), horizon_accepted=sum(bool(r['horizon_found']) for r in rows[12]),
    final_mass=last['mass'], final_expansion_rms=last['horizon_trial_expansion_rms'],
    last_accepted_mass_row=next((r for r in reversed(rows[12]) if r['horizon_found'] and r['mass'] is not None), None),
    exact_time_pairs=pairs,
    qualification='Both terminal states admitted and preserved. Same radial18 configuration, fixed coordinate cutoff r>.3, N12/N16 and dt refined together. N16 covers only .6M; no higher-resolution evidence beyond that time. Constraint-residual order, not field self-convergence or 10M proof. Missing horizon masses retained; no interpolation or smoothing.')
(O / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['last_M', 'fixed03', 'horizon_rows', 'horizon_accepted', 'final_mass', 'final_expansion_rms']}))
