#!/usr/bin/env python3
"""Spatial self-convergence at common saved CFL-aligned physical times.

No interpolation in time. Independent constraints remain all-step/stage monitors.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from numerics import weights
from kernel import PARITY
from trajectory import read_trajectory,prefix_stage_extrema
from analyze_principal_controls import fit_order,norm
p=argparse.ArgumentParser();inputs=p.add_mutually_exclusive_group(required=True);inputs.add_argument('--input',type=Path)
inputs.add_argument('--case-paths',type=Path,nargs=3);p.add_argument('--output',type=Path,required=True)
p.add_argument('--resolutions',type=int,nargs=3,default=[512,1024,2048]);p.add_argument('--interval',type=float,default=.1);p.add_argument('--start',type=float,default=0.);p.add_argument('--end',type=float);p.add_argument('--retained-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
runs=[]
for case_index,n in enumerate(a.resolutions):
 folder=a.input/f'spatial-n{n}' if a.input else a.case_paths[case_index];m,rows,stages,segments=read_trajectory(folder,allow_domain_transfer=a.retained_only)
 if m['arguments']['n']!=n:raise ValueError('Resolution label differs from source-bound input')
 if m['status']!='finished':raise ValueError('Require completed trajectories')
 if a.end is not None:
  if a.end>m['time']+1e-10:raise ValueError('Prefix end exceeds completed trajectory')
  stages=prefix_stage_extrema(rows,a.end)
 profiles={}
 for segment in segments:
  for path in sorted(Path(segment['folder']).glob('profile-*.npz')):
   with np.load(path,allow_pickle=False) as z:
    t=float(z['t']);tick=int(round(t/a.interval))
    if abs(t-tick*a.interval)>1e-10:continue
    q=z['q_physical' if 'q_physical' in z else 'q'].copy();r=z['r'].copy();frame_step=int(z['step']) if 'step' in z else None
    if 'time_lattice' in z:
        epoch=json.loads(str(z['time_lattice']));represented=epoch['origin_time']+(frame_step-epoch['origin_step'])*epoch['dt']
        if abs(t-represented)>1e-10:raise ValueError('Saved frame differs from its timestep lattice')
   if tick in profiles and not np.array_equal(profiles[tick]['q'],q):raise ValueError('Duplicate continuation landing states differ')
   profiles[tick]=dict(t=t,step=frame_step,r=r,q=q,path=str(path),sha256=sha(path))
 runs.append(dict(n=n,manifest=m,stages=stages,segments=segments,profiles=profiles))
if a.end is None and np.ptp([r['manifest']['time'] for r in runs])>1e-10:raise ValueError('Different final times')
if len({r['manifest']['arguments']['cfl'] for r in runs})!=1:raise ValueError('Spatial study requires a fixed CFL')
if any(z['manifest']['binding']!=runs[0]['manifest']['binding'] for z in runs):raise ValueError('Spatial study requires identical numerical source and binaries')
end=int(round((runs[0]['manifest']['time'] if a.end is None else a.end)/a.interval));start_tick=int(round(a.start/a.interval))
if start_tick<0 or end<=start_tick or abs(start_tick*a.interval-a.start)>1e-10:raise ValueError('Invalid exact analysis window')
requested=list(range(start_tick,end+1))
missing={z['n']:sorted(set(requested)-set(z['profiles'])) for z in runs}
ticks=sorted(set(requested)&set.intersection(*[set(z['profiles']) for z in runs]))
if len(ticks)<3 or ticks[0]!=start_tick or ticks[-1]!=end:raise ValueError('Common profiles do not cover the trajectory interval')
hs=[float(np.diff(z['profiles'][start_tick]['r'])[0]) for z in runs];errors={key:[] for key in ['bulk','endpoint']}
for lo,hi in zip(runs[:-1],runs[1:]):
 coarse=lo['profiles'][start_tick]['r'];fine=hi['profiles'][start_tick]['r'];width=9
 ratio=(len(fine)-1)//(len(coarse)-1)
 nested=coarse[0]==fine[0] and ratio*(len(coarse)-1)==len(fine)-1
 if nested:
  ratio=(len(fine)-1)//(len(coarse)-1)
  if ratio*(len(coarse)-1)!=len(fine)-1 or not np.array_equal(coarse,fine[::ratio]):raise ValueError('Nodal study requires exactly nested grids')
 elif coarse[0]<=0 or fine[0]<=0:raise ValueError('Unsupported inner grid')
 ext=np.r_[-fine[:width][::-1],fine];indices=[];coefficients=[]
 for x in coarse:
  start=min(max(np.searchsorted(ext,x)-width//2,0),len(ext)-width);idx=np.arange(start,start+width)
  indices.append(idx);coefficients.append(weights(ext[idx],x))
 indices=np.array(indices);coefficients=np.array(coefficients);series={key:[] for key in errors}
 for tick in ticks:
  q=hi['profiles'][tick]['q'];parity=np.r_[PARITY,np.ones(max(0,q.shape[1]-len(PARITY)))];values=np.r_[q[:width][::-1]*parity,q]
  sampled=q[::ratio] if nested else np.einsum('ij,ijf->if',coefficients,values[indices]);delta=lo['profiles'][tick]['q']-sampled
  for key,mask in [('bulk',coarse<.8),('endpoint',coarse>.9)]:series[key].append(norm(delta,coarse,mask))
 for key in errors:errors[key].append(series[key])
result={}
times=np.array(ticks)*a.interval
for region,series in errors.items():
 maximum=[float(max(e)) for e in series];l2=[float(np.sqrt(np.sum(np.diff(times)*(np.array(e[:-1])**2+np.array(e[1:])**2)/2)/(times[-1]-times[0]))) for e in series]
 result[region]=dict(sampled_differences=series,maximum_differences=maximum,maximum_order=fit_order(maximum,hs),
  time_RMS_differences=l2,time_RMS_order=fit_order(l2,hs),sampled_orders=[fit_order([e[i] for e in series],hs) for i in range(len(ticks))])
record=dict(scope='Physical fields on the explicitly recorded time window at common exact saved times; no temporal interpolation. Nested nodal grids use exact restriction; regular half-cell grids use degree8 spatial interpolation. AF node has zero radial norm weight and is not a physical metric sample. Field errors supplement all-stage safety and every-accepted-step constraints.',
 analysis_window=dict(start=a.start,end=end*a.interval,retained_only=a.retained_only,actual_trajectory_ends=[z['manifest']['time'] for z in runs],stage_scope='Recorded cumulative stages through exact report at end' if a.end is not None else 'Entire completed trajectory'),
 interval=a.interval,times=times.tolist(),h=hs,regions=result,missing_requested_ticks=missing,
 coverage_note='Only common reported-time matches within1e-10 are compared. Older cumulative-clock frames can miss a requested tick by one RK step; none are shifted or interpolated in time. The integer-step clock repair is a separate source-bound change.',
 runs=[dict(n=z['n'],binding=z['manifest']['binding'],segments=z['segments'],all_stage_extrema=z['stages'],
       selected_profiles=[{key:value for key,value in z['profiles'][i].items() if key not in ['q','r']} for i in ticks]) for z in runs],
 source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('analyze_principal_controls.py'),Path(__file__).with_name('trajectory.py')]})
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for ax,region in zip(axes,['bulk','endpoint']):
 for pair,e in zip(zip(a.resolutions[:-1],a.resolutions[1:]),errors[region]):ax.semilogy(record['times'],e,label=f'{pair[0]} to {pair[1]}')
 ax.set_title(region+': physical field differences');ax.set_xlabel('Coordinate time');ax.legend();ax.grid(alpha=.25)
fig.savefig(a.output/'trajectory.png',dpi=160);fig.savefig(a.output/'trajectory.pdf');plt.close(fig)
print(json.dumps({key:{k:v for k,v in value.items() if not k.startswith('sampled_')} for key,value in result.items()}))
