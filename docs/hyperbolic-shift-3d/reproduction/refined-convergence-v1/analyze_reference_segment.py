"""Qualified first-segment physics/cost summary; keep terminal checkpoint separate."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;O=D/'stationary-reference-v2-segment01';raw=O/'output'
rows=[json.loads(x) for x in (raw/'state.history.jsonl').read_text().splitlines()];report=json.loads((raw/'state.report.json').read_text());term=json.loads((O/'terminal-verification.json').read_text());assert term['admitted'] and report['checkpoint_admitted'] and term['checkpoint']['steps']==rows[-1]['step']==12753
phases={};nested={};boundary=[]
for p in raw.glob('state.rank*.performance.json'):
 x=json.loads(p.read_text())
 for name,v in x['exclusive_phases'].items():phases.setdefault(name,[]).append(v['seconds'])
for p in raw.glob('state.rank*.generations.jsonl'):
 x=json.loads(p.read_text().splitlines()[-1]);boundary.append(x['boundary_subphases'])
 for name,v in x['nested_seconds'].items():nested.setdefault(name,[]).append(v)
summary=lambda values:dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values)))
good=[x for x in rows if x['horizon_found'] and x['mass'] is not None];err=np.array([100*(x['mass']/(1/3)-1) for x in good]);t=np.array([x['t_over_M'] for x in rows])
r=dict(status='admitted_healthy_walltime_not_accuracy_success',last_M=rows[-1]['t_over_M'],steps=rows[-1]['step'],online_accepted=len(good),rows=len(rows),max_abs_mass_error_percent=float(max(abs(err))),final_mass_error_percent=float(err[-1]),final_fixed03=rows[-1]['fixed'][1],final_fixed06=rows[-1]['fixed'][2],cumulative_counters={k:v for k,v in report.items() if k.startswith('cumulative_')},native_command_seconds=json.loads((raw/'worker.json').read_text())['native_seconds'],exclusive_rank_seconds={k:summary(v) for k,v in phases.items()},step_seconds_per_call=summary(np.array(phases['step'])/report['steps']),nested_rank_seconds={k:summary(v) for k,v in nested.items()},boundary_subphase_rank_seconds=[summary(np.array(boundary)[:,k]) for k in range(6)],qualification='Admitted checkpoint at4.782375M, not10M or convergence. Fixed coordinate norms outside every complete candidate enclosure; not invariant physical domains. Counter arrays and scri maxima persist through RF2 restart: final cumulative totals or differences, never sum successive cumulative reports. Boundary counters separate. Host timers have no added fences; nested timers overlap and must not be summed as exclusive phases. No controlled scaling/speedup claim.')
(O/'history-cost-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
for ax,f in zip(axes[0],('H','M')):
 for label,y in [('full',[x[f] for x in rows]),('candidate enclosure',[x[f+'_exterior'] for x in rows]),('fixed r > .3',[x['fixed'][1][f] for x in rows]),('fixed r > .6',[x['fixed'][2][f] for x in rows])]:ax.semilogy(t,y,'.-',label=label,ms=3)
 ax.set(xlabel='t/M',ylabel=f+' constraint RMS');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[1,0].semilogy([x['t_over_M'] for x in good],np.maximum(abs(err),1e-16),'.-',label='accepted |mass error| [%]');axes[1,0].semilogy(t,[x['horizon_trial_expansion_rms'] for x in rows],'.-',label='candidate expansion RMS');axes[1,0].axhline(.001,color='gray',ls=':',label='online expansion tolerance');axes[1,0].set(xlabel='t/M',ylabel='Horizon diagnostics');axes[1,0].legend(fontsize=8);axes[1,0].grid(alpha=.2)
for field,label in [('null','surface null'),('ghost_null','ghost null'),('pole_max','cumulative pole'),('clock_max','cumulative clock')]:axes[1,1].semilogy(t,[x[field] for x in rows],'.-',label=label,ms=3)
axes[1,1].set(xlabel='t/M',ylabel='Residual');axes[1,1].legend(fontsize=8);axes[1,1].grid(alpha=.2)
fig.suptitle('Reference-v2 N16: healthy walltime checkpoint at4.7824M; continuation submitted')
fig.savefig(O/'histories.png',dpi=170);fig.savefig(O/'histories.pdf');plt.close(fig)
print(json.dumps({k:r[k] for k in ('last_M','online_accepted','rows','max_abs_mass_error_percent','final_mass_error_percent','step_seconds_per_call','nested_rank_seconds')},indent=2))
