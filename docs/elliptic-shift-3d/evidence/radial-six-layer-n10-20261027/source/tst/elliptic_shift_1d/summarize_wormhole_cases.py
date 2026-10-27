#!/usr/bin/env python3
"""Preserve source-bound wormhole outcomes without equating survival with success."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
records=[];fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for run in a.run:
    m=json.loads((run/'manifest.json').read_text())
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    if not history:continue
    label=run.name;t=np.array([row['t'] for row in history]);final=history[-1]
    peaks={}
    for row in history:
        for key,value in row.get('all_RK_stages',{}).items():
            peaks[key]=(min if key.startswith('min_') else max)(peaks.get(key,value),value)
    mass=m['arguments']['mass']
    record=dict(run=str(run),status=m['status'],arguments=m['arguments'],binding=m['binding'],
        final_time=final['t'],all_stage_extrema=peaks,
        final={k:final.get(k) for k in ['H_rms','M_rms','Theta_rms','Z_rms','mass_r0.5',
            'apparent_horizon_mass','AF_inner_mass','AF_shift_slope','inner_areal_radius',
            'inner_radius_log_slope','inner_proper_distance_log_slope','inner_physical_lapse_log_slope']},
        completed_requested_time=m['status']=='finished',
        pole_component_pass=peaks.get('max_scri_pole',float('inf'))<=1e-7,
        trumpet_result_admitted=False)
    if 'failure' in m:record['failure']=m['failure']
    records.append(record)
    axes[0,0].semilogy(t,np.maximum([row['H_rms'] for row in history],1e-25),label=label)
    axes[0,1].semilogy(t,np.maximum([row['M_rms'] for row in history],1e-25),label=label)
    axes[1,0].plot(t,[row.get('apparent_horizon_mass',np.nan)/mass-1 for row in history],label=label)
    axes[1,1].semilogy(t,np.maximum([row.get('all_RK_stages',{}).get('max_scri_pole',row['scri_pole']) for row in history],1e-25),label=label)
for ax,title in zip(axes.ravel(),['Hamiltonian RMS','Momentum RMS','Horizon mass / initial M - 1','Cumulative all-stage scri pole']):
    ax.set_title(title);ax.set_xlabel('coordinate time');ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8);axes[1,1].axhline(1e-7,color='k',linestyle=':',label='declared component gate')
fig.savefig(a.output/'review.png',dpi=180);fig.savefig(a.output/'review.pdf');plt.close(fig)
record=dict(cases=records,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    interpretation='Constraint and invariant diagnostics must converge and the inner geometry must settle before any wormhole-to-trumpet admission. Every plotted survival/failure remains an unaccepted control unless separately assessed.')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps([dict(name=Path(row['run']).name,status=row['status'],t=row['final_time'],
    H=row['final']['H_rms'],M=row['final']['M_rms'],pole=row['all_stage_extrema'].get('max_scri_pole')) for row in records]))
