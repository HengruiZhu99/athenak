"""Admitted same-grid timestep sensitivity through .6M; no time-order claim."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;O=D/'time-control-comparison';O.mkdir(exist_ok=True)
folders=[D/'stationary-reference-v2-segment01',D/'time-control-segment01'];histories=[];hashes=[]
for F in folders:
    term=json.loads((F/'terminal-verification.json').read_text());assert term['admitted']
    local=json.loads((F/'local-verification.json').read_text());assert local['all_worker_artifacts_verified']
    p=F/'output/state.history.jsonl';hashes.append(hashlib.sha256(p.read_bytes()).hexdigest());histories.append([json.loads(x) for x in p.read_text().splitlines() if json.loads(x)['t_over_M']<=.6+1e-10])
analysis=json.loads((folders[1]/'analysis-command.json').read_text());assert analysis['checkpoint_unchanged'] and analysis['returncode']==0 and max(x for row in analysis['native_report_norms_match'] for x in row)<1e-9
baseline,small=histories;matched=[]
for b in baseline:
    match=[x for x in small if abs(x['time']-b['time'])<1e-10];assert len(match)==1;x=match[0]
    assert all(q['horizon_complete'] and q['horizon_max_radius']<.3 for q in (b,x))
    matched.append(dict(t_over_M=b['t_over_M'],max_time_mismatch_M=abs(b['t_over_M']-x['t_over_M']),relative_fixed03_change={f:x['fixed'][1][f]/b['fixed'][1][f]-1 for f in ('H','M','Mr')},baseline_fixed03=b['fixed'][1],smaller_dt_fixed03=x['fixed'][1]))
report=json.loads((folders[1]/'output/state.report.json').read_text());worker=json.loads((folders[1]/'output/worker.json').read_text())
r=dict(status='both_controls_admitted',history_sha256=hashes,baseline_dt=.000125,smaller_dt=.00003125,matched_recorded_times=matched,max_abs_relative_fixed03_change={f:max(abs(x['relative_fixed03_change'][f]) for x in matched) for f in ('H','M','Mr')},smaller_dt_rows=len(small),smaller_dt_online_accepted=sum(x['horizon_found'] for x in small),smaller_dt_native_seconds=worker['native_seconds'],smaller_dt_counters={k:v for k,v in report.items() if k.startswith('cumulative_')},qualification='Same radial12 mesh, geometry, gauge, fixed effective boundary penalty and native operator. Fixed r>.3 outside complete recorded candidate enclosures. Exact recorded-time matches, no interpolation/smoothing. Diagnostic cadence differs but does not modify evolved fields. Fourfold dt reduction barely changes exterior residuals through.6M; this does not prove a spatial cause, time-order, long-time accuracy or asymptotic convergence. Baseline terminal counters belong to4.782375M and are not compared to.6M totals.')
(O/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
for rows,label in [(baseline,'dt = 1.25e-4'),(small,'dt = 3.125e-5')]:
    common=[x for x in rows if any(abs(x['t_over_M']-m['t_over_M'])<1e-10 for m in matched)]
    for ax,f in zip(axes[0],('H','M')):ax.semilogy([x['t_over_M'] for x in common],[x['fixed'][1][f] for x in common],'.-',label=label,ms=4)
for ax,f in zip(axes[0],('H','M')):ax.set(xlabel='t/M',ylabel=f+' RMS, fixed r > .3');ax.legend()
for ax,f in zip(axes[1],('H','M')):
    ax.plot([x['t_over_M'] for x in matched],[100*x['relative_fixed03_change'][f] for x in matched],'.-');ax.set(xlabel='t/M',ylabel=f+' relative change [%]')
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('Same-grid exterior norms: matched recorded times, 4× smaller timestep',fontsize=12)
fig.savefig(O/'histories.png',dpi=170);fig.savefig(O/'histories.pdf');plt.close(fig)
print(json.dumps(r,indent=2))
