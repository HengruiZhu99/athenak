"""Plot retained histories only. Gaps remain gaps; protocols are labeled."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;REP=HERE.parent
assert json.loads((HERE/'n24-failure/verification-and-location.json').read_text())['verified_failure_evidence']
assert json.loads((HERE/'n16-completed/checkpoint-verified.json').read_text())['completed']
paths={
 'fresh16':HERE/'n16-completed/puncture.history.csv',
 'fresh24':HERE/'n24-failure/puncture.history.csv',
 'smooth_log16':REP/'smooth-gauge-20m-v1/n16-failure/puncture.history.csv',
 'smooth_log16_early':REP/'smooth-gauge-v1/n16.history.csv',
 'smooth_log24':REP/'smooth-gauge-v1/n24.history.csv',
 'original16':REP/'resolution-20m-v1/all-available-history/n16-merged.csv',
 'original24':REP/'resolution-20m-v1/all-available-history/n24-merged.csv',
 'one_d':REP/'original-1d-reference/history.csv'}
data={k:np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True)) for k,p in paths.items()}
audit=json.loads((REP/'aurora/stability-puncture-v1/verification.json').read_text())['original_1d_reference']
assert hashlib.sha256(paths['one_d'].read_bytes()).hexdigest()==audit['history_sha256']
# Join identical smooth-log stages, never bridging absent diagnostics.
a=data['smooth_log16_early'];b=data['smooth_log16'];data['smooth_log16']=np.concatenate([a[a['t_over_M']<b['t_over_M'][0]-1e-9],b])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={16:'#b45418',24:'#1769aa'}
fig,axes=plt.subplots(2,2,figsize=(12,8))
for j,(key,ref,title) in enumerate([('H_exterior_L2','H_exterior_rms','Hamiltonian'),('Mr_exterior_L2','M_exterior_rms','Radial momentum')]):
 for row,end in enumerate([20,3.8]):
  ax=axes[row,j]
  for n in [16,24]:
   d=data[f'fresh{n}'];ax.plot(d['t_over_M'],d[key],'-o',ms=3,color=colors[n],label=f'3D N{n}: direct from t=0')
  d=data['one_d'];m=d['t_over_M']<=end;ax.plot(d['t_over_M'][m],d[ref][m],color='black',ls=':',lw=1.6,label='Preserved 1D N513')
  ax.axvline(3.726,color=colors[24],ls='--',lw=.9,alpha=.6)
  ax.set(xlim=(0,end),ylim=(0,None),title=title+(' — full history' if row==0 else ' — early evolution'),xlabel='t / M',ylabel='Exterior volume RMS');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('Fresh direct-variable smoother-gauge runs: N16 completes 20M; N24 fails at 3.726M',fontsize=13)
fig.subplots_adjust(top=.90,bottom=.15,hspace=.35,wspace=.28)
fig.text(.5,.045,'N24 diagnostics end at 3.6M; dashed line marks its last accepted time. No missing-value interpolation.\n3D exterior cuts use complete horizon candidates; finder flags are preserved. 1D is much finer and uses different numerics.\nThese data do not demonstrate convergence; each run uses its own horizon cutoff.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(HERE/f'constraints-final.{ext}',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for j,n in enumerate([16,24]):
 for i,(key,title) in enumerate([('H_exterior_L2','Hamiltonian'),('Mr_exterior_L2','Radial momentum')]):
  ax=axes[i,j]
  for prefix,label,color,style in [('fresh','Smooth blend; direct from t=0',colors[n],'-'),('smooth_log','Smooth blend; log variables','#a55295','--'),('original','Original blend; log → direct at 3M','#777777',':')]:
   d=data[f'{prefix}{n}'];ax.plot(d['t_over_M'],d[key],style,color=color,label=label,linewidth=1.5)
  ax.set(xlim=(0,20 if n==16 else 6),ylim=(0,None),title=f'N{n}: {title}',xlabel='t / M',ylabel='Exterior volume RMS');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Protocol comparisons: changing variables from the start changes the trajectory',fontsize=14)
fig.subplots_adjust(top=.90,bottom=.13,hspace=.35,wspace=.28)
fig.text(.5,.04,'Retained samples only; gaps are not filled. Earlier runs differ in variable chart, floors and/or gauge blend.\nSmooth-log N24 shown only for its completed matched 3M trial; original long-run horizons retain their quality gaps.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(HERE/f'protocol-comparison.{ext}',dpi=170)
summary={n:dict(last_diagnostic_t_over_M=float(data[f'fresh{n}']['t_over_M'][-1]),diagnostic_rows=len(data[f'fresh{n}']),accepted_horizons=int((data[f'fresh{n}']['horizon_found']==1).sum()),candidate_horizons=int((data[f'fresh{n}']['horizon_complete_surface']==1).sum())) for n in [16,24]}
(HERE/'comparison-sources.json').write_text(json.dumps(dict(coverage=summary,sources={str(p.relative_to(REP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},scope='Unequal 1D/3D resolution and differing historical protocols; no convergence order fitted.'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
