#!/usr/bin/env python3
"""Summarize a frozen live snapshot; never reads or changes the remote job."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def hist(path):
    text=path.read_text()
    header=next(s for s in text.splitlines() if '[1]=' in s)
    columns=re.findall(r'\[\d+\]=([^\s]+)',header)
    a=np.loadtxt(path,ndmin=2)
    assert a.shape[1]==len(columns) and np.isfinite(a).all()
    assert np.all(np.diff(a[:,0])>=0)
    return dict(zip(columns,a.T))

def main():
    snapshot=json.loads((HERE/'snapshot.json').read_text())
    u=hist(HERE/'pulse/ks_background.user.hst')
    c=hist(HERE/'pulse/ks_background.z4c.user.hst')
    zero=hist(HERE/'zero/ks_background.z4c.user.hst')
    gate=json.loads((HERE/'zero/run-validation.json').read_text())
    log=(HERE/'pulse/run.log').read_text()
    assert np.all(u['bad-metric']==0)
    assert gate['passed'] and gate['residual_exactly_zero'] and gate['ranks']==24
    p=json.loads((HERE/'theta-profiles.json').read_text())
    records=re.findall(r'elapsed=([\d.eE+\-]+) cycle=(\d+) time=([\d.eE+\-]+) dt=([\d.eE+\-]+)',log)
    records=np.array(records,dtype=float)
    last=records[-1];start=records[-min(6,len(records))]
    seconds_per_M=(last[0]-start[0])/(last[2]-start[2])
    summary=dict(collected_utc=snapshot['collected_utc'],scheduler=snapshot['status'],
                 current_status='running; no final pulse validation or completion claim',
                 history_end_M=float(u['time'][-1]),zero_gate=dict(time_M=gate['time_M'],ranks=gate['ranks'],blocks=gate['blocks'],exact_zero=gate['residual_exactly_zero'],all_payload_finite=gate['all_payload_finite'],invalid_metric_cells_including_ghosts=gate['invalid_metric_cells_including_ghosts'],source='zero/run-validation.json'),
                 all_collected_histories_finite=True,bad_metric_max=float(max(u['bad-metric'])),
                 throughput=dict(window_M=[float(start[2]),float(last[2])],seconds_per_M=float(seconds_per_M),seconds_per_cycle=float(seconds_per_M*last[3]),dt_M=float(last[3]),time_to_1000M_from_last_record_hours=float((1000-last[2])*seconds_per_M/3600),remaining_time_estimate_scope='Extrapolation of the recent measured window, not a target-completion prediction within this PBS allocation'),
                 amplitudes={},raw_constraint_integral_comparison={},theta_profiles=dict(cohorts=len(p['included']),first_M=p['included'][0]['time_M'],last_M=p['included'][-1]['time_M'],excluded=p['excluded'],scope=p['scope']),limitations=['Raw full-background H and M have nonzero discretization baselines. Differences of scalar norms are not norms of a residual constraint field.','This short pulse response and its moving maxima do not establish an instability origin or long-term stability.','Theta binary dumps are float32 active cells; ghost-metric validity is checked separately at the completed 50M checkpoint only.'])
    checkpoint=json.loads((HERE/'pulse-checkpoint-50M.json').read_text())
    assert checkpoint['passed'] and checkpoint['ranks']==24 and checkpoint['cycle']==2000
    summary['pulse_checkpoint_50M']={key:checkpoint[key] for key in ['passed','time_M','cycle','ranks','blocks','all_payload_finite','invalid_metric_cells_including_ghosts','minimum_raw_full','collected_utc']}
    summary['pulse_checkpoint_50M']['source']='pulse-checkpoint-50M.json'
    for name in ['Theta-max','alpha-res','beta-res','Gam-res']:
        j=int(np.argmax(u[name]))
        summary['amplitudes'][name]=dict(initial=float(u[name][0]),final=float(u[name][-1]),max=float(u[name][j]),time_of_max_M=float(u['time'][j]))
    for name in ['H-norm2','M-norm2','H-int2','M-int2']:
        baseline=float(zero[name][0]);assert c[name][0]==baseline
        relative=c[name]/baseline-1
        summary['raw_constraint_integral_comparison'][name]=dict(initial_zero_gate_baseline=baseline,final=float(c[name][-1]),final_relative_change=float(relative[-1]),max_abs_relative_change=float(np.max(abs(relative))))
    summary['final_proper_exterior_rms']={name:float(np.sqrt(c[key][-1]/c['Volume'][-1])) for name,key in [('Theta','Theta-norm'),('H','H-norm2'),('M','M-norm2'),('Z','Z-norm2')]}
    hmax=re.findall(r'Z4C_EXT_HMAX time=(\S+) H=(\S+) x=(\S+) y=(\S+) z=(\S+) r_bh=(\S+)',log)
    summary['logged_raw_exterior_Hmax']={'first':dict(zip(['time_M','H','x','y','z','radius_M'],map(float,hmax[0]))),'last':dict(zip(['time_M','H','x','y','z','radius_M'],map(float,hmax[-1]))),'scope':'Raw full constraint maxima are dominated by initial background truncation near r=1.06617M; not the peak of H(t)-H(0).'}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':9,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for name in ['Theta-max','alpha-res','beta-res','Gam-res']:
        axes[0,0].semilogy(u['time'],np.where(u[name]>0,u[name],np.nan),label=name)
    axes[0,0].set_ylabel('Active maximum absolute residual')
    axes[0,0].legend(fontsize=8)
    for name,key in [('Theta','Theta-norm'),('Z','Z-norm2')]:
        a=np.sqrt(c[key]/c['Volume']);axes[0,1].semilogy(c['time'],np.where(a>0,a,np.nan),label=name)
    axes[0,1].set_ylabel('Exterior proper-volume RMS');axes[0,1].legend()
    for name in ['H-norm2','M-norm2']:
        axes[1,0].plot(c['time'],(c[name]/c[name][0]-1)*1e6,label=name)
    axes[1,0].set_ylabel('Raw constraint integral change (ppm)');axes[1,0].legend()
    times=[s['time_M'] for s in p['included']]
    for name,label in [('protected_r_le_8','r ≤ 8'),('ramp_8_lt_r_lt_28','8 < r < 28'),('full_sponge_r_ge_28','r ≥ 28'),('outer_face_distance_le_2','Face distance ≤ 2')]:
        a=np.array([s['regions'][name]['coordinate_rms'] for s in p['included']]);axes[1,1].semilogy(times,np.where(a>0,a,np.nan),label=label)
    axes[1,1].set_ylabel('Regional coordinate-volume Theta RMS');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('t / M');ax.grid(alpha=.2)
    fig.savefig(HERE/'early-response.png',dpi=160)
    files=[q for q in HERE.rglob('*') if q.is_file() and q.name!='manifest.json']
    (HERE/'manifest.json').write_text(json.dumps({str(q.relative_to(HERE)):{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(files)},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
