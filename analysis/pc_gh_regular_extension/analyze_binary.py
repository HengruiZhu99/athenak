"""Audit new binary experiments against matched and saved controls.

No completion marker implies physics qualification. Preserve every finite
history sample and distinguish incomplete merger windows from measured results.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_native_puncture import cells, symmetry
from make_inputs import parse
from read_complete_history import read_history

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/pc_gh_bbh'))
from plot_long_comparison import load_tracker, load_wave, load_amr_changes
sys.path.insert(0,str(ROOT/'analysis/pc_gh_localization'))
from plot_qualification import regional_rms


def ratio_rms(numerator,volume):
    if np.any(numerator<0) or np.any(volume<=0):
        raise ValueError('Invalid norm numerator or volume')
    return np.sqrt(numerator/volume)


def history(run,z4c=False):
    h,audit=read_history(next(run.glob('*.z4c.user.hst' if z4c else '*.pcgh.hst')))
    if z4c:
        series=dict(time=h['time'],H=ratio_rms(h['H-norm2'],h['Volume']),
                    M=ratio_rms(h['M-norm2'],h['Volume']))
    else:
        series=dict(time=h['time'],H=ratio_rms(h['shared-H-n'],h['shared-Vol']),
                    M=ratio_rms(h['shared-M-n'],h['shared-Vol']))
        series.update(regional_rms(h,'chi'))
    if not all(np.isfinite(v).all() for v in series.values()):
        raise ValueError(f'Nonfinite normalized history: {run}')
    return series,audit


def native(run,tracks,params):
    result=[]
    for path in sorted((run/'bin').glob('*.pcgh.[0-9]*.bin')):
        data,xyz,dx,u=cells(path);time=float(data['time'])
        if u.shape[1]!=55 or not np.isfinite(u).all():
            raise ValueError(f'Invalid native state: {path}')
        centers=[]
        for hole,track in enumerate(tracks):
            if time==0:
                centers.append([float(params['pc_gh'][f'co_{hole}_{d}']) for d in ['x','y','z']])
                continue
            if time<track['time'][0]-1e-6 or time>track['time'][-1]+1e-6:
                raise ValueError('Native output time is outside the copied tracker interval')
            centers.append([float(np.interp(time,track['time'],track[d])) for d in ['x','y','z']])
        per_hole=[]
        for center in centers:
            radius=np.linalg.norm(xyz-center,axis=1)
            closest=int(np.argmin(radius))
            per_hole.append(dict(center=center,closest_sample=xyz[closest].tolist(),
                distance=float(radius[closest]),spacing=dx[closest].tolist(),
                rho=float(u[closest,18]),w=float(u[closest,0]),
                L=float(np.linalg.norm(u[closest,43:46])),Q=float(np.linalg.norm(u[closest,25:43]))))
        parity=symmetry(xyz,u,data['var_names'],operations=('reflect_x','reflect_y'))
        result.append(dict(time=time,source_file=str(path.resolve()),
            max_abs_by_component=dict(zip(data['var_names'],map(float,np.abs(u).max(axis=0)))),
            symmetry=parity,punctures=per_hole))
    return result


def audit_run(run,include_native=True):
    params=parse((run/'used_input.athinput').read_text())
    s,audit=history(run)
    bounds=np.atleast_1d(np.genfromtxt(next(run.glob('*.pcgh-boundedness.dat')),names=True))
    if not all(np.isfinite(bounds[n]).all() for n in bounds.dtype.names):
        raise ValueError(f'Nonfinite uncensored bounds: {run}')
    failures=[]
    for log in sorted(run.glob('segment-*.log')):
        failures.extend(dict(log=log.name,line=line) for line in log.read_text().splitlines()
                        if 'FATAL ERROR' in line)
    tracks=[load_tracker(next(run.glob(f'*.co_{i}.txt'))) for i in [0,1]]
    if not np.array_equal(tracks[0]['time'],tracks[1]['time']):
        raise ValueError('Paired trackers have different times')
    record=dict(run=str(run),parameters=params,history_audit=audit,
        completed=(run/'completed.json').exists(),failures=failures,
        history_time=float(s['time'][-1]),bounds_time=float(bounds['time'][-1]),
        bounds_final={n:float(bounds[n][-1]) for n in bounds.dtype.names},
        bounds_min={n:float(bounds[n].min()) for n in bounds.dtype.names},
        bounds_max={n:float(bounds[n].max()) for n in bounds.dtype.names},
        constraints_final={n:float(v[-1]) for n,v in s.items() if n!='time'},
        tracker_symmetry=dict(max_x_pair_sum=float(np.abs(tracks[0]['x']+tracks[1]['x']).max()),
            max_transverse=float(max(np.abs(t[d]).max() for t in tracks for d in ['y','z'])),
            time=float(tracks[0]['time'][-1]),final_positions=[[float(t[d][-1]) for d in ['x','y','z']] for t in tracks]),
        amr_changes=load_amr_changes(run),
        provenance=json.loads((run/'provenance.json').read_text()),native=[])
    if include_native: record['native']=native(run,tracks,params)
    return record,s,tracks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',type=Path,nargs='+')
    ap.add_argument('--z4c',type=Path,required=True)
    ap.add_argument('--projected',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--skip-native',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[];series={};track_data={};directories={}
    for run in args.runs:
        record,s,tracks=audit_run(run,not args.skip_native)
        rate=record['parameters']['pc_gh']['reduction_rate']
        order=record['parameters']['pc_gh']['spatial_order']
        label=f'FD{order}, lambda={rate}'
        if label in series: raise ValueError('Duplicate experiment label')
        records.append(record);series[label]=s;track_data[label]=tracks;directories[label]=run
        print(label,'t=',record['history_time'],'completed=',record['completed'],
              'failed=',bool(record['failures']),'H/M=',s['H'][-1],s['M'][-1])
    for label,run,z4c in [('Saved Z4c',args.z4c,True),('Saved projected PC-GH',args.projected,False)]:
        series[label],_=history(run,z4c);directories[label]=run
    colors={label:color for label,color in zip(series,plt.rcParams['axes.prop_cycle'].by_key()['color'])}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for label,s in series.items():
        for ax,key in zip(axes.flat,['H','M','reduction','curl']):
            if key in s: ax.plot(s['time'],s[key],label=label,color=colors[label])
    for ax,key in zip(axes.flat,['H','M','reduction','curl']):
        ax.set(xlabel='t/M',ylabel=key+' RMS',yscale='log')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Head-on constraints: shared ADM norms; PC-GH reductions/curls use chi selection\nSaved controls differ in discretization and GH damping')
    fig.savefig(args.output/'constraints.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for label,tracks in track_data.items():
        for n,t in enumerate(tracks):
            ax.plot(t['time'],t['x'],ls='-' if n==0 else '--',label=label if n==0 else None,color=colors[label])
    for i in [0,1]:
        t=load_tracker(next(args.z4c.glob(f'*.co_{i}.txt')))
        ax.plot(t['time'],t['x'],color='k',lw=.9,ls='-' if i==0 else '--',label='Saved Z4c' if i==0 else None)
    ax.set(xlabel='t/M',ylabel='Puncture x/M',title='ODE puncture tracks, with restart segments merged')
    ax.grid(alpha=.2);ax.legend();fig.savefig(args.output/'tracks.png',dpi=180);plt.close(fig)
    windows={};fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    for radius,ax in zip([8,12,24,32,48,56],axes.flat):
        windows[str(radius)]={}
        for label,run in directories.items():
            wave=load_wave(run,radius);t=wave['retarded_time'];real=wave['real_22'];imag=wave['imag_22']
            ax.plot(t,real,label=label,lw=1,color=colors[label])
            mask=(t>=20)&(t<=35)
            entry=dict(available_retarded_interval=[float(t[0]),float(t[-1])],
                complete_merger_window=bool(t[0]<=20 and t[-1]>=35))
            if entry['complete_merger_window']:
                energy=np.trapezoid(real[mask]**2,t[mask])
                entry.update(peak_real_abs=float(np.abs(real[mask]).max()),
                    imaginary_to_real_l2=float(np.sqrt(np.trapezoid(imag[mask]**2,t[mask])/energy)) if energy else None)
            windows[str(radius)][label]=entry
        ax.set(xlabel='u = (t-r)/M',ylabel='Re(r Psi4 [2,2])',title=f'r={radius}M')
        ax.axvspan(20,35,color='grey',alpha=.08);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Uncensored available waveforms; shaded window is the saved merger interval\nIncomplete windows are not assigned merger amplitudes')
    fig.savefig(args.output/'waveforms.png',dpi=180);plt.close(fig)
    report=dict(runs=records,wave_windows=windows,
        scope='Research comparison; clean exit alone is not qualification. Native bounds are slice-only.',
        saved_references=dict(z4c=str(args.z4c),projected_pcgh=str(args.projected)))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
