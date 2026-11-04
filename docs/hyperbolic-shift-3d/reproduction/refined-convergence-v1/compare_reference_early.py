"""Immutable early live snapshot vs preserved failed constant-driver control."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists());DOC=Path(__file__).resolve().parent
OUT=DOC/'stationary-reference-v2-early';OUT.mkdir(exist_ok=True)
snapshot=json.loads((ROOT/'output/refined-convergence/reference-v2-live-snapshot.json').read_text());raw=snapshot.pop('history');worker=snapshot.pop('worker')
assert len(raw.encode())==snapshot['history_bytes'] and hashlib.sha256(raw.encode()).hexdigest()==snapshot['history_sha256']
assert snapshot['plan_sha256']==json.loads((DOC/'stationary-reference-v2-submission/submission.json').read_text())['plan_sha256']
new=[json.loads(x) for x in raw.splitlines()];assert new[-1]['t_over_M']==snapshot['last_M']
old_path=ROOT/'output/refined-convergence/frame-separation-aurora/campaigns/stationary20-constant-n16-segment01/output/state.history.jsonl';old_raw=old_path.read_bytes();old=[json.loads(x) for x in old_raw.decode().splitlines()]
worker_old=json.loads((old_path.parent/'worker.json').read_text());assert hashlib.sha256(old_raw).hexdigest()==worker_old['artifacts']['state.history.jsonl']['sha256']
(OUT/'reference-history.jsonl').write_text(raw);(OUT/'reference-worker-live.json').write_text(worker);(OUT/'baseline-history.jsonl').write_bytes(old_raw)
common=[]
for a in old:
 b=next((b for b in new if b['step']==a['step'] and abs(b['time']-a['time'])<1e-12),None)
 if b is not None:
  common.append(dict(step=a['step'],t_over_M=a['t_over_M'],original_r06=a['fixed'][2],reference_r06=b['fixed'][2],original_pole=a['pole_max'],reference_pole=b['pole_max'],original_clock=a['clock_max'],reference_clock=b['clock_max']))
snapshot.update(common=common,baseline_job='9038285',baseline_history_sha256=hashlib.sha256(old_raw).hexdigest(),qualification='Live reference snapshot, not terminal admission. Original failed at .463875M and is never restarted. Fixed coordinate cuts and diagnostic operators matched, but gauge evolution differs; cumulative pole/clock maxima are not instantaneous values. No interpolation, extrapolated missing masses, or convergence claim.')
(OUT/'comparison.json').write_text(json.dumps(snapshot,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
for rows,label,color in ((old,'Original outer gauge','tab:orange'),(new,'Stationary-reference v2','tab:blue')):
 t=[x['t_over_M'] for x in rows]
 for ax,f in zip(axes[0],('H','M')):ax.semilogy(t,[x['fixed'][2][f] for x in rows],'.-',color=color,label=label);ax.set(title=f'{f} outside fixed r > .6',xlabel='t/M',ylabel='Constraint RMS')
 for field,style in (('pole_max','-'),('clock_max','--')):axes[1,0].semilogy(t,[x[field] for x in rows],'.'+style,color=color,label=label+' '+field.split('_')[0])
 good=[x for x in rows if x['horizon_found'] and x['mass'] is not None];err=[abs(100*(x['mass']/(1/3)-1)) for x in good];axes[1,1].semilogy([x['t_over_M'] for x in good],err,'.-',color=color,label=label)
axes[1,0].set(title='Cumulative scri maxima',xlabel='t/M',ylabel='Residual maximum');axes[1,1].set(title='Online accepted horizon masses only',xlabel='t/M',ylabel='Absolute mass error [%]')
for ax in axes.flat:ax.axvline(.463875,color='gray',ls=':',lw=1);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=7)
fig.suptitle(f"Early live snapshot through {new[-1]['t_over_M']:.2f}M; dotted line: old failure")
fig.savefig(OUT/'early-comparison.png',dpi=170);fig.savefig(OUT/'early-comparison.pdf');plt.close(fig)
print(json.dumps(dict(last_M=snapshot['last_M'],common_last=common[-1]),indent=2))
