#!/usr/bin/env python3
"""Lapse status figures from immutable saved times, explicitly allowing a partial fine run.

This is an illustrative status report, never a convergence/admission assessor.
"""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_telegraph_time import numerical_binding
from telegraph_analysis_io import validate_telegraph_profile

p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
times=[0.,1.,2.5,4.,6.,8.,10.];runs={};records=[];binding=None;configuration=None
for n in [1025,2049,4097]:
 folder=a.matrix/f'n{n}';m=json.loads((folder/'manifest.json').read_text())
 if 'checkpoint_parent' in m:raise ValueError('This status plot expects initial-to-time runs, not checkpoint segments')
 if m['status'] not in ['running','finished','failed']:raise ValueError('Run has not initialized')
 actual_binding=numerical_binding(folder.resolve(),m)
 config={k:v for k,v in m['arguments'].items() if k not in ['n','output','resume','library','fast_library']}
 if binding is None:binding=actual_binding;configuration=config
 if binding!=actual_binding or configuration!=config:raise ValueError('Uncontrolled sources or numerical inputs')
 data={};receipts=[]
 for t in times:
  if t>m['time']+1e-12:continue
  index=round(t/m['arguments']['report_dt'])+1
  path=folder/f'profile-{index:04d}.npz'
  with np.load(path,allow_pickle=False) as z:
   if abs(float(z['t'])-t)>1e-12:raise ValueError('Selected report is not at the requested exact time')
   d={key:z[key].copy() for key in ['r','grid_jacobian','q','q_physical','diagnostic','B','beta','gauge_speeds','rays']}
  validate_telegraph_profile(d);r=d['r'];O=(1-r*r)/2;A=(1+r*r)/2
  alpha=A*r*r/(r*r+O**4)*np.exp(d['q'][:,6])
  np.testing.assert_allclose(alpha,d['diagnostic'][:,7],atol=1e-14,rtol=2e-13)
  physical=np.divide(alpha,O,out=np.full_like(alpha,np.nan),where=O>0)
  data[t]=dict(r=r,alpha=alpha,physical=physical,H=d['diagnostic'][:,0])
  receipts.append(dict(time=t,path=str(path.resolve()),sha256=sha(path)))
 runs[n]=data
 (a.output/f'manifest-n{n}-at-capture.json').write_text(json.dumps(m,indent=2)+'\n')
 records.append(dict(n=n,status_at_capture=m['status'],available_through=m['time'],source_binding=actual_binding,selected_profiles=receipts,
                     source=str(folder.resolve()),all_RK_stages_at_capture=m['all_RK_stages']))
if 10. not in runs[2049] or 2.5 not in runs[4097]:raise ValueError('Need completed middle grid and a fine snapshot at t2.5')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'mathtext.fontset':'stix'})
colors={0.:'#777777',1.:'#4477aa',2.5:'#228899',4.:'#228833',6.:'#aa8800',8.:'#cc6677',10.:'#aa3377'}
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for t in times:
 d=runs[2049][t];axes[0].plot(d['r'],d['alpha'],color=colors[t],label=f't={t:g}',lw=1.5)
 if t>=4:
  mask=(d['r']>=1e-8)&(d['r']<=.03)
  axes[1].semilogx(d['r'][mask],d['physical'][mask],color=colors[t],label=f't={t:g}',lw=1.6)
axes[0].set(title='Global conformal lapse',xlabel='Compactified radius r',ylabel=r'$\alpha$',xlim=(0,1));axes[0].legend(ncol=2,fontsize=8)
axes[1].set(title='Physical lapse in the shrinking inner region',xlabel='Compactified radius r (log scale)',ylabel=r'$\widetilde\alpha=\alpha/\Omega$',xlim=(1e-8,.03));axes[1].legend(fontsize=8)
for ax in axes:ax.set_ylim(bottom=0);ax.grid(alpha=.2)
fig.suptitle('Lapse evolution • N=2049 • completed through t=10',fontsize=13)
for ext in ['png','pdf']:fig.savefig(a.output/('lapse-evolution.'+ext))
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,6.8),gridspec_kw={'height_ratios':[2.5,1]},layout='constrained')
styles={1025:dict(color='#cc6677',ls='--',lw=1.5),2049:dict(color='#228899',ls='-',lw=1.7),4097:dict(color='#332288',ls=':',lw=1.8)}
metrics=[]
for col,(t,limits) in enumerate([(2.5,(.03,.22)),(10.,(1e-8,3e-5))]):
 available=[n for n in sorted(runs) if t in runs[n]]
 for n in available:
  d=runs[n][t];mask=(d['r']>=limits[0])&(d['r']<=limits[1]);axes[0,col].plot(d['r'][mask],d['physical'][mask],label=f'N={n}',**styles[n])
 for n,m in zip(available,available[1:]):
  coarse,fine=runs[n][t],runs[m][t];ratio=(m-1)//(n-1)
  if not np.array_equal(coarse['r'],fine['r'][::ratio]):raise ValueError('Resolution profiles do not share physical nodes')
  mask=(coarse['r']>=limits[0])&(coarse['r']<=limits[1]);delta=coarse['physical'][mask]-fine['physical'][::ratio][mask]
  axes[1,col].plot(coarse['r'][mask],delta,label=f'{n} − {m}',color=styles[n]['color'],lw=1.2)
  metrics.append(dict(t=t,grids=[n,m],radius_window=list(limits),max_lapse_difference=float(np.max(abs(delta)))))
 axes[0,col].set_title(f'Physical lapse at t={t:g}'+(' • finest grid pending' if len(available)<3 else ' • three resolutions'))
 axes[0,col].set_ylabel(r'$\widetilde\alpha$');axes[0,col].legend(fontsize=8)
 axes[1,col].set(xlabel='Compactified radius r'+(' (log scale)' if col else ''),ylabel=r'$\Delta\widetilde\alpha$')
 axes[1,col].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[1,col].legend(fontsize=8)
 for ax in axes[:,col]:
  ax.set_xlim(*limits);ax.grid(alpha=.2)
  if col:ax.set_xscale('log')
 axes[0,col].tick_params(labelbottom=False)
fig.suptitle('Resolution comparison • raw saved profiles, without smoothing',fontsize=13)
for ext in ['png','pdf']:fig.savefig(a.output/('lapse-resolution.'+ext))
plt.close(fig)
result=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,selected_profile_differences=metrics,
 method='Alpha independently reconstructed from the AF logarithmic state and checked against the saved lapse. Physical lapse is alpha/Omega; the global plot uses conformal alpha because physical lapse diverges at scri. Profiles use actual grid nodes with no smoothing or temporal interpolation.',
 scope='Updated plots only. N4097 may be partial; only saved times at or before its captured progress are shown. t10 three-grid convergence and independent quarter-step/robustness controls remain unverified. Nonmonotone late lapse structure is shown explicitly, not declared eliminated.',goal_status='incomplete')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=[{k:r[k] for k in ['n','status_at_capture','available_through']} for r in records],profile_differences=metrics),indent=2))
