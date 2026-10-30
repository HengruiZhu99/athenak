#!/usr/bin/env python3
"""Compare the bounded smooth-weight trial to the original measured replay."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
RAW=ROOT/'output/hyperbolic-core-shell/smooth-gauge-v1/aurora';OLD=HERE.parent/'resolution-20m-v1/candidate-history'
verification=json.loads((RAW/'local-verification.json').read_text());assert verification['verified']
keys=['H_exterior_L2','Mr_exterior_L2','M_exterior_L2','Theta_exterior_L2'];data={};reports={};sources={}
for variant in ['old','new']:
 for n in [16,24]:
  p=(OLD/f'n{n}.history.csv') if variant=='old' else (RAW/f'output/n{n}.history.csv')
  rows=list(csv.DictReader(p.open()))
  if variant=='old' and n==16:
   early=HERE.parent/'n16-timestep-to3m-v1/n16-original-dt-reproduction-v1/output/puncture-initial.history.csv';rows=list(csv.DictReader(early.open()))+rows;sources[str(early)]=hashlib.sha256(early.read_bytes()).hexdigest()
  unique={round(float(r['t_over_M']),9):r for r in rows};rows=[unique[t] for t in sorted(unique)]
  a={k:np.array([float(r[k]) for r in rows]) for k in ['t_over_M']+keys};data[variant,n]=a
  assert a['t_over_M'][0]==0 and a['t_over_M'][-1]==3
  rp=(OLD/f'n{n}.report.json') if variant=='old' else RAW/f'output/n{n}.report.json'
  report=json.loads(rp.read_text());assert report['status']=='completed' and report['checkpoint_admitted'];reports[variant,n]=report
  if variant=='new':
   assert report['gauge_blend']=='smoothstep9_w_056_080' and report['checkpoint_schema']=='HYP3D_MPI_GW1'
   (HERE/f'n{n}.history.csv').write_bytes(p.read_bytes());(HERE/f'n{n}.report.json').write_bytes(rp.read_bytes())
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False});colors={16:'#b45418',24:'#1769aa'}
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for ax,k,title in zip(axes.flat,keys,['Hamiltonian','Radial momentum','Conformal invariant momentum','Z4 scalar Theta']):
 for variant,style in [('old','--'),('new','-')]:
  for n in [16,24]:
   a=data[variant,n];ax.plot(a['t_over_M'],a[k],style,color=colors[n],lw=1.7,marker='o' if variant=='new' else None,ms=3,label=f'N{n} '+('original' if variant=='old' else 'smooth blend'))
 ax.set(title=title,ylabel='Exterior volume RMS',xlim=(0,3),ylim=(0,None));ax.grid(alpha=.2)
for ax in axes[1]:ax.set_xlabel('t/M')
axes[0,0].legend(fontsize=9,ncol=2);fig.suptitle('Broader gauge blend: matched N16 / N24 evolution to 3M',fontsize=16)
fig.subplots_adjust(top=.90,bottom=.14,hspace=.27,wspace=.25)
fig.text(.5,.035,'Changed w(r) affects gauge coupling and Z4 damping. Same initial data, grid, dt, log variables and boundary coefficients.\nComplete horizon candidates define each run’s exterior cutoff; acceptance flags are preserved. No interpolation of missing values.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(HERE/f'constraint-history-comparison.{ext}',dpi=180)
radial={};summary={};common=None
for variant in ['old','new']:
 for n in [16,24]:
  d=json.loads((HERE/f'radial/{variant}-n{n}.json').read_text());radial[variant,n]=d
  if common is None:common=d['cutoff']
  assert d['cutoff']==common and d['t_over_M']==3
for n in [16,24]:
 r=reports['new',n];summary[n]=dict(new_status=r['status'],max_null_surface=r['max_null_surface'],max_null_ghost=r['max_null_ghost'],horizon_found=r.get('horizon_found'),horizon_candidate=r.get('horizon_trial_radius'),horizon_expansion=r.get('horizon_trial_expansion_rms'),common_cutoff=common,old=radial['old',n]['norms']['exterior'],new=radial['new',n]['norms']['exterior'])
 summary[n]['new_over_old']={k:summary[n]['new'][k]/summary[n]['old'][k] for k in ['H_L2','Mr_L2','M_L2','Theta_L2']}
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,k,title in zip(axes,['H2','Mr2'],['Hamiltonian band RMS at 3M','Radial momentum band RMS at 3M']):
 for variant,style in [('old','--'),('new','-')]:
  for n in [16,24]:
   c=radial[variant,n]['radial_cumulative'];r=np.array([v['cutoff'] for v in c]);volume=np.array([v['volume'] for v in c]);a=np.array([v[k] for v in c]);dv=volume[:-1]-volume[1:];assert np.all(dv>0);v=np.sqrt(np.maximum(0,a[:-1]-a[1:])/dv)
   ax.stairs(v,r,baseline=None,color=colors[n],linestyle=style,label=f'N{n} {variant}')
 ax.axvspan(.6,.8,color='#e9bf36',alpha=.13);ax.axvline(.75,color='.5',ls=':');ax.set(title=title,xlim=(common,1),yscale='log',xlabel='Compactified radius r',ylabel='Band RMS');ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(HERE/f'radial-comparison.{ext}',dpi=180)
regions={}
for variant in ['old','new']:
 for n in [16,24]:
  d=radial[variant,n];c={v['cutoff']:v for v in d['radial_cumulative']};regions[f'{variant}-n{n}']={}
  for lo,hi in [(.16,.18),(.4,.6),(.4,.675),(.6,.8),(.675,.75)]:
   dv=c[lo]['volume']-c[hi]['volume'];row={}
   for k in ['H2','Mr2','M2']:
    integral=c[lo][k]-c[hi][k]
    row[k]={'rms':float(np.sqrt(max(0,integral)/dv)),'exterior_squared_integral_fraction':integral/c[common][k]}
   regions[f'{variant}-n{n}'][f'{lo}-{hi}']=row
ratios={variant:{k:radial[variant,24]['norms']['exterior'][k]/radial[variant,16]['norms']['exterior'][k] for k in ['H_L2','Mr_L2','M_L2','Theta_L2']} for variant in ['old','new']}
(HERE/'comparison.json').write_text(json.dumps(dict(summary=summary,radial_regions=regions,n24_over_n16=ratios,sources=sources),indent=2)+'\n');print(json.dumps(dict(summary=summary,n24_over_n16=ratios),indent=2))
