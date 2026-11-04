"""Join admitted reference segments without interpolation or counter double counting."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
O=D/'stationary-reference-v2-completed';O.mkdir(exist_ok=True)
rows=[];segments=[];inputs={}
for number in (1,2):
    S=D/f'stationary-reference-v2-segment{number:02d}'
    t=json.loads((S/'terminal-verification.json').read_text());assert t['admitted']
    report=json.loads((S/'output/state.report.json').read_text())
    h=S/'output/state.history.jsonl';inputs[str(h.relative_to(D))]=hashlib.sha256(h.read_bytes()).hexdigest()
    chunk=[json.loads(x) for x in h.read_text().splitlines()]
    duplicate=0
    for x in chunk:
        if rows and x['step']==rows[-1]['step']:
            assert abs(x['time']-rows[-1]['time'])<1e-14
            duplicate+=1;continue
        assert not rows or x['time']>rows[-1]['time']
        rows.append(x)
    timing=[]
    for p in (S/'output').glob('state.rank*.performance.json'):
        timing.append(json.loads(p.read_text())['exclusive_phases']['step']['seconds'])
    worker=json.loads((S/'output/worker.json').read_text())
    segments.append(dict(number=number,end_step=report['steps'],end_M=report['t_over_M'],duplicate_boundary_rows=duplicate,native_command_seconds=worker['native_seconds'],median_step_seconds=float(np.median(timing)),counters={k:v for k,v in report.items() if k.startswith('cumulative_')}))
assert rows[-1]['step']==26667 and rows[-1]['t_over_M']==10
good=[x for x in rows if x['horizon_found'] and x['mass'] is not None]
err=np.array([100*(x['mass']/(1/3)-1) for x in good]);t=[x['t_over_M'] for x in rows]
summary=dict(status='admitted_completed_10M_not_convergence',inputs=inputs,segments=segments,rows=len(rows),online_accepted=len(good),final_fixed03=rows[-1]['fixed'][1],final_mass_error_percent=float(err[-1]),max_abs_mass_error_percent=float(max(abs(err))),final_horizon_rms=rows[-1]['horizon_trial_expansion_rms'],native_command_seconds=sum(s['native_command_seconds'] for s in segments),step_seconds_per_step=sum(s['median_step_seconds'] for s in segments)/26667,cumulative_counters=segments[-1]['counters'],floor_counters_unchanged_after_segment01=segments[0]['counters']==segments[1]['counters'],max_complete_horizon_radius=max(x['horizon_max_radius'] for x in rows if x['horizon_complete']),qualification='Native online acceptance retained. Final cumulative counters, not sum of restart reports. Fixed coordinate r>.3 outside complete horizon enclosures; not an invariant physical domain. Raw sampling times joined without interpolation; restart sampling phase retained. Native commands include setup/diagnostics and exclude queue time. No controlled strong-scaling or convergence claim.')
(O/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
(O/'history.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
for ax,f in zip(axes[0],('H','M')):
    for label,y in [('full',[x[f] for x in rows]),('candidate exterior',[x[f+'_exterior'] for x in rows]),('fixed r > .3',[x['fixed'][1][f] for x in rows])]:ax.semilogy(t,y,'.-',label=label,ms=3)
    ax.set(xlabel='t/M',ylabel=f+' constraint RMS');ax.legend(fontsize=8)
axes[1,0].semilogy([x['t_over_M'] for x in good],np.maximum(abs(err),1e-16),'.-',label='accepted |mass error| [%]')
axes[1,0].semilogy(t,[x['horizon_trial_expansion_rms'] for x in rows],'.-',label='expansion RMS')
axes[1,0].set(xlabel='t/M',ylabel='Horizon diagnostics');axes[1,0].legend(fontsize=8)
for field,label in [('null','surface null'),('ghost_null','ghost null'),('pole_max','cumulative pole'),('clock_max','cumulative clock')]:axes[1,1].semilogy(t,[x[field] for x in rows],'.-',label=label,ms=3)
axes[1,1].set(xlabel='t/M',ylabel='Residual');axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.2);ax.axvline(segments[0]['end_M'],color='gray',ls=':',lw=.8)
fig.suptitle('Reference-v2 stationary N16: completed 10M; exterior error remains\nDotted line: verified restart boundary, no interpolated history samples')
fig.savefig(O/'histories.png',dpi=170);fig.savefig(O/'histories.pdf');plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','segments')},indent=2))
