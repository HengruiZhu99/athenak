#!/usr/bin/env python3
"""Plot every retained exterior diagnostic; never fill missing measurements."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
REP=BASE.parent
sources={16:[REP/'n16-timestep-to3m-v1/n16-original-dt-reproduction-v1/output/puncture-initial.history.csv', REP/'direct-chi-lapse-v1/segment01/puncture.history.csv',BASE/'candidate-history/n16.history.csv'],24:[BASE/'n24-failure/puncture.history.csv',BASE/'candidate-history/n24.history.csv'],32:[BASE/'n32-at3m/puncture.history.csv']}
if (HERE/'n32-live.history.csv').exists():sources[32].append(HERE/'n32-live.history.csv')
keys=['H_exterior_L2','Mr_exterior_L2','M_exterior_L2','Theta_exterior_L2']
data={};audit={};colors={16:'#b45418',24:'#1769aa',32:'#278047'}
for n,paths in sources.items():
 rows={}
 for p in paths:
  for row in csv.DictReader(p.open()):
   t=round(float(row['t_over_M']),9)
   if t not in rows or sum(np.isfinite(float(row[k])) for k in keys)>=sum(np.isfinite(float(rows[t][k])) for k in keys):rows[t]=row
 rows=[rows[t] for t in sorted(rows)];data[n]={k:np.array([float(r[k]) for r in rows]) for k in ['t_over_M']+keys}
 with (HERE/f'n{n}-merged.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 audit[n]={'last_diagnostic_t_over_M':float(data[n]['t_over_M'][-1]),'rows':len(rows),'finite_exterior_rows':int(np.isfinite(data[n][keys[0]]).sum()),'sources':{str(p.relative_to(REP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
labels=[r'$\|H\|_{2,\rm ext}$',r'$\|M_r\|_{2,\rm ext}$',r'$\|M\|_{2,\rm ext}$',r'$\|\Theta\|_{2,\rm ext}$']
for suffix,xmax in [('full',20),('early',6)]:
 fig=plt.figure(figsize=(13,9));gs=fig.add_gridspec(3,2,height_ratios=[1,1,.25],hspace=.38,wspace=.28)
 axes=[fig.add_subplot(gs[i//2,i%2]) for i in range(4)]
 for ax,k,label,title in zip(axes,keys,labels,['Hamiltonian','Radial momentum','Conformal invariant momentum','Z4 scalar']):
  for n,d in data.items():ax.plot(d['t_over_M'],d[k],'-o',color=colors[n],ms=3,lw=1.5,label=f'N{n}')
  ax.axvline(3,color='.4',ls=':',lw=1)
  ax.axvline(5.869,color=colors[24],ls='--',lw=1,alpha=.6)
  ax.set(xlim=(0,xmax),ylim=(0,None),title=title,ylabel=label,xlabel=r'$t/M$');ax.grid(alpha=.2)
 axes[0].legend(ncol=3,loc='upper right')
 coverage=fig.add_subplot(gs[2,:]);coverage.set(xlim=(0,xmax),yticks=[0,1,2],yticklabels=['N16','N24','N32'],ylim=(-.5,2.5),xlabel=r'$t/M$ — saved exterior-diagnostic coverage')
 for i,n in enumerate([16,24,32]):
  d=data[n];finite=np.isfinite(d[keys[0]]);end={16:20,24:5.869,32:float(d['t_over_M'][-1])}[n]
  coverage.plot([0,end],[i,i],color=colors[n],alpha=.25,lw=3)
  coverage.plot(d['t_over_M'][finite],np.full(finite.sum(),i),'|',color=colors[n],ms=11)
  coverage.plot(d['t_over_M'][~finite],np.full((~finite).sum(),i),'x',color='.55',ms=3)
 coverage.grid(axis='x',alpha=.15)
 fig.suptitle('Exterior constraint histories: all retained N16 / N24 / N32 measurements',fontsize=16,y=.98)
 fig.text(.5,.94,'N16 completed 20M; N24 failed after 5.869M; N32 continuation is running (snapshot).',ha='center',fontsize=10)
 fig.subplots_adjust(top=.875,bottom=.14)
 fig.text(.5,.035,'Dotted line: log → direct variables at 3M. Dashed blue line: N24 failure. Gaps / gray ×: exterior norm not recorded.\nN16/N24 through 3M use complete candidate cutoffs; otherwise only accepted horizons. No missing-value interpolation.\nNormalized compactified-coordinate volume RMS; each run uses its own horizon radius.',ha='center',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(HERE/f'exterior-constraints-{suffix}.{ext}',dpi=180)
(HERE/'sources-and-coverage.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
