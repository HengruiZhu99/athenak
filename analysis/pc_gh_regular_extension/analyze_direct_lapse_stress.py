"""Compare the frozen R16 stress baseline and direct-gradient correction.

Inputs contain native histories, fatal/segment logs and the compact monitor
outputs from reduce_direct_lapse_monitor.py. No simulation is launched here.
"""
import argparse
import json
import re
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from read_complete_history import read_history
from make_inputs import parse


def norms(h,region):
    if region=='all':
        v=h['all-Vol']; keys=['all-'+x for x in ['Cp2','Z2','H2','M2','rw2','rQ2','ra2','rB2','cp2','cQ2','cL2','cB2']]
    else:
        v=h['Volume'];keys=['Cperp-n2','Z-norm2','H-norm2','Mhat-norm2','redw-norm2','redQ-norm2','reda-norm2','redB-norm2','curlp-n2','curlQ-n2','curlL-n2','curlB-n2']
    a=[h[k] for k in keys]
    result={'GH':np.sqrt((a[0]+a[1])/v),'H':np.sqrt(a[2]/v),'M':np.sqrt(a[3]/v),
            'reduction':np.sqrt(sum(a[4:8])/v),'curl':np.sqrt(sum(a[8:12])/v)}
    for k,val in zip(['Rw','RQ','Ralpha','RB','curl_p','curl_Q','curl_L','curl_B'],a[4:]):
        result[k]=np.sqrt(val/v)
    return result


def load(run):
    h,audit=read_history(next(run.glob('*.pcgh.hst')))
    params=parse((run/'used_input.athinput').read_text())
    fatal=[line for f in sorted(run.glob('segment-*.log')) for line in f.read_text().splitlines() if 'FATAL ERROR' in line]
    m=pd.read_csv(run/'completed-monitor.csv.gz')
    m=m.drop_duplicates(['cycle','t_step','region','quantity'],keep='last')
    m['time']=m.t_step+m.dt
    # A final callback can carry a last dt after the endpoint; retain the raw row
    # but do not mislabel it as another evolved step past the requested target.
    m=m[m.time<=float(params['time']['tlim'])+1e-10]
    if not np.isfinite(m[['time','max','coordinate_l1']].to_numpy()).all():
        raise ValueError('Nonfinite completed monitor')
    stage=pd.read_csv(run/'stage-envelopes.csv.gz')
    status=dict(path=str(run),history_end=float(h['time'][-1]),fatal_errors=fatal,
                completed=(run/'completed.json').exists(),history_audit=audit,
                monitor_audit=json.loads((run/'monitor-audit.json').read_text()),
                minimum_completed={},minimum_stage={},peak_completed={},late_growth_flags=[])
    for q in ['min_eigenvalue','min_w','min_rho','min_alpha']:
        for data,key in [(m,'minimum_completed'),(stage,'minimum_stage')]:
            rows=data[(data.region=='all')&(data.quantity==q)]
            status[key][q]=rows.loc[rows['max'].idxmin()].to_dict()
    for (region,q),rows in m.groupby(['region','quantity']):
        if q.startswith('min_'):continue
        peak=rows.loc[rows['max'].idxmax()].to_dict();status['peak_completed'][region+':'+q]=peak
        bins=rows.groupby(np.floor(rows.time*2).astype(int))['max'].max()
        doublings=[int(bins.index[i]) for i in range(3,len(bins))
                   if np.all(bins.iloc[i-2:i+1].to_numpy()>2*bins.iloc[i-3:i].to_numpy())]
        ref=rows[(rows.time>=2)&(rows.time<3)]['max'].max()
        late=rows[rows.time>=max(3.,rows.time.max()-1.)]
        ratio=late['max'].max()/ref if ref>0 else None
        slope=None
        if len(late)>2 and (late['max']>0).all():slope=float(np.polyfit(late.time,np.log(late['max']),1)[0])
        if doublings or (ratio is not None and ratio>10 and slope is not None and slope>0):
            status['late_growth_flags'].append(dict(region=region,quantity=q,doubling_bins=doublings,
                late_over_2to3_peak=ratio,late_log_slope=slope))
    return h,m,stage,status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    runs=[a.baseline,a.corrected];loaded=[load(r) for r in runs]
    labels=['Saved factorized target','Direct product gradient']; colors=['#737373','#0072B2']
    params=[parse((r/'used_input.athinput').read_text()) for r in runs]
    if params[0]!=params[1]:raise ValueError('Stress input parameters differ')
    summary=dict(runs=[x[3] for x in loaded],input_parameters_equal=True,
        input_sha256=[hashlib.sha256((r/'used_input.athinput').read_bytes()).hexdigest() for r in runs],
        common_time_comparison={},scope=__doc__)
    common_end=min(x[0]['time'][-1] for x in loaded)
    times=np.arange(.5,common_end+1e-9,.5)
    for region in ['all','chi']:
        summary['common_time_comparison'][region]={}
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for ax,q in zip(axes.flat,['GH','H','M','reduction','curl']):
            record=[]
            for (h,m,stage,status),label,color in zip(loaded,labels,colors):
                norm=norms(h,region)[q];ax.semilogy(h['time'],norm,label=label,color=color)
                record.append(dict(times=times.tolist(),rms=np.interp(times,h['time'],norm).tolist(),
                    final_time=float(h['time'][-1]),final_rms=float(norm[-1]),peak_rms=float(norm.max())))
            summary['common_time_comparison'][region][q]=record
            ax.set(title=q,xlabel='t/M',ylabel='Coordinate-volume RMS');ax.grid(alpha=.25)
        axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'R16, finest M/256\nIdentical input; target 6M\nchi threshold = 0.0625\nReductions retain historical Ralpha',va='top')
        axes.flat[0].legend(fontsize=8);fig.suptitle('Full domain' if region=='all' else 'Chi-excised constraints')
        fig.savefig(a.output/f'{region}-constraints.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for row,region in enumerate(['all','chi']):
        for ax,q in zip(axes[row],['curl_p','curl_Q','curl_L','curl_B']):
            for (h,*_),label,color in zip(loaded,labels,colors):
                ax.semilogy(h['time'],norms(h,region)[q],label=label,color=color)
            ax.set(title=f'{region}: {q}',xlabel='t/M',ylabel='Coordinate-volume RMS');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.savefig(a.output/'individual-curls.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,q in zip(axes.flat,['RQ','curl_Q','curl_L','RL_direct','min_eigenvalue','min_rho']):
        for (_,m,_,_),label,color in zip(loaded,labels,colors):
            rows=m[(m.region=='all')&(m.quantity==q)].sort_values('time')
            if rows.empty:continue
            ax.plot(rows.time,rows['max'],label=label,color=color)
        if not q.startswith('min_'):ax.set_yscale('log')
        ax.set(title=q,xlabel='t/M',ylabel='Full-domain extremum');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Completed-step extrema; uncensored by chi')
    fig.savefig(a.output/'health-and-growth.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    m=loaded[1][1]
    for ax,q in zip(axes.flat,['RQ','curl_Q','curl_L','RL_direct','abs_H','min_eigenvalue']):
        for region in ['core','taper','exterior','interface_faces']:
            rows=m[(m.region==region)&(m.quantity==q)].sort_values('time')
            if not rows.empty:ax.plot(rows.time,rows['max'],label=region)
        if not q.startswith('min_'):ax.set_yscale('log')
        ax.set(title=q,xlabel='t/M',ylabel='Regional extremum');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Direct gradient: growth localization (overlapping interface stratum)')
    fig.savefig(a.output/'localization.png',dpi=170);plt.close(fig)
    corrected=summary['runs'][1]
    if corrected['fatal_errors']:decision='FAIL'
    elif not corrected['completed'] or corrected['history_end']<6-1e-8:decision='INCONCLUSIVE'
    else:decision='REQUIRES_REVIEW'
    summary['gate2_mechanical_decision']=decision
    summary['gate2_note']='No automatic science pass. Inspect regional and late growth before promotion.'
    (a.output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(decision=decision,history_ends=[x[3]['history_end'] for x in loaded],
                         fatal_errors=corrected['fatal_errors'],flags=corrected['late_growth_flags']),indent=2))

if __name__=='__main__':main()
