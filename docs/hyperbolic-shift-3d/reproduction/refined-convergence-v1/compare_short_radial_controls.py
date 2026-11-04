"""Compare admitted time/radial controls to the preserved long reference history."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;O=D/'short-radial-controls-comparison';O.mkdir(exist_ok=True)
cases={};inputs={}
for key,folder,dt in [('reference','stationary-reference-v2-segment01',.000125),('time','time-control-segment01',.00003125),('radial','radial-control-segment01',.00003125)]:
    S=D/folder;t=json.loads((S/'terminal-verification.json').read_text());assert t['admitted']
    if key!='reference':
        a=json.loads((S/'analysis-command.json').read_text());assert a['checkpoint_unchanged'] and a['returncode']==0
        assert max(x for row in a['native_report_norms_match'] for x in row)<1e-9
    h=S/'output/state.history.jsonl';inputs[key]=hashlib.sha256(h.read_bytes()).hexdigest()
    allrows=[json.loads(s) for s in h.read_text().splitlines()];rows=[x for x in allrows if x['t_over_M']<=.6+1e-10]
    report=json.loads((S/'output/state.report.json').read_text());assert report['requested_dt']==dt
    assert abs(rows[-1]['t_over_M']-.6)<1e-10
    assert all(x['horizon_complete'] and x['horizon_max_radius']<.3 for x in rows)
    assert all(abs(x['penalty']-.6510416666660156)<1e-12 for x in rows)
    cases[key]=dict(folder=S,rows=rows,report=report,dt=dt)
matches=[]
for base in cases['reference']['rows']:
    pair={}
    for key in ('time','radial'):
        found=[x for x in cases[key]['rows'] if abs(x['t_over_M']-base['t_over_M'])<1e-10]
        assert len(found)==1;pair[key]=found[0]
    item=dict(time_M=base['t_over_M'],max_time_mismatch_M=max(abs(x['t_over_M']-base['t_over_M']) for x in pair.values()))
    for f in ('H','M','Mr'):
        values=[x['fixed'][1][f] for x in (base,pair['time'],pair['radial'])]
        item[f]=dict(reference=values[0],time=values[1],radial=values[2],time_over_reference=values[1]/values[0],radial_over_time=values[2]/values[1])
    matches.append(item)
summary=dict(history_sha256=inputs,common_recorded_times=matches,case_summaries={})
for key in ('time','radial'):
    c=cases[key];rows=c['rows'];report=c['report'];worker=json.loads((c['folder']/'output/worker.json').read_text());sp=json.loads((c['folder']/'spatial-native.json').read_text());bins=np.array(sp['radial_bins']);integ=np.array(sp['integrals']);band=(bins[:,0]>=.9-1e-12)&(bins[:,1]<=.95+1e-12)
    summary['case_summaries'][key]=dict(dt=c['dt'],online_accepted=sum(x['horizon_found'] for x in rows),rows=len(rows),final_mass=rows[-1]['mass'],final_fixed03=rows[-1]['fixed'][1],native_seconds=worker['native_seconds'],floor_and_sign_counters={k:v for k,v in report.items() if k.startswith('cumulative_')},band_090_095={f:dict(squared_integral=float(bins[band,i].sum()),exterior_fraction=float(bins[band,i].sum()/integ[1,i-2])) for f,i in [('H',3),('M',4),('Mr',5)]})
summary['qualification']='Admitted .6M controls, same native operator and fixed coordinate r>.3 outside complete candidate enclosures. Match recorded times only; no interpolation/smoothing. Reference/time isolate dt; time/radial change radial resolution AND interface placement, holding final scri shell/core/angles/geometry/gauge/effective penalty fixed. No long-time, uniform-resolution, field-self-convergence or fourth-order proof. Cumulative counters shown per fresh case, never summed between cases.'
(O/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
labels=dict(reference='radial12, dt=1.25e-4',time='radial12, dt=3.125e-5',radial='radial18, dt=3.125e-5')
for key,c in cases.items():
    for ax,f in zip(axes,('H','M')):ax.semilogy([x['t_over_M'] for x in c['rows']],[x['fixed'][1][f] for x in c['rows']],'.-',label=labels[key],ms=4)
for ax,f in zip(axes,('H','M')):ax.set(xlabel='t/M',ylabel=f+' RMS, fixed r > .3');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Bounded time/radial attribution controls; no interpolation');fig.savefig(O/'histories.png',dpi=170);fig.savefig(O/'histories.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for key in ('time','radial'):
    b=np.array(json.loads((cases[key]['folder']/'spatial-native.json').read_text())['radial_bins']);mask=b[:,0]>=.3-1e-12
    for ax,i in zip(axes,(3,4)):ax.semilogy((b[mask,0]+b[mask,1])/2,b[mask,i],'.-',label=labels[key],ms=4)
for ax,f in zip(axes,('H','M')):ax.set(xlabel='Compact r',ylabel=f+' squared integral per .01 bin');ax.axvspan(.9,.95,color='gray',alpha=.1);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Same-dt admitted .6M endpoints: native radial error integrals');fig.savefig(O/'radial-errors.png',dpi=170);fig.savefig(O/'radial-errors.pdf');plt.close(fig)
print(json.dumps(summary,indent=2))
