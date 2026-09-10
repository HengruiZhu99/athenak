from pathlib import Path
import sys,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910');sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
from criterion import read_history
cases=[]
for n in [128,256,512]:
 d=Path(f'/pscratch/sd/h/hzhu/n{n}-failed-amplitude-20260910') if n!=256 else Path('/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910/cycle_04')
 cases.extend([('dchi',n,d),('chiTE',n,root/'production'/f'N{n}')])
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');fig2,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');summary=[]
colors={128:'#2878b5',256:'#df7e21',512:'#31924c'}
for method,n,d in cases:
 files=list(d.glob('*.hst'))
 if not files:continue
 h=read_history(files[0]);t=np.array([x['time'] for x in h]);rc=(d/'run-status').read_text().strip() if (d/'run-status').exists() else None
 label=f'N{n} '+('running' if rc is None else ('failed' if rc!='0' else 'stopped'))
 row=0 if method=='dchi' else 1
 for ax,key,ylabel in zip(axs[row],['C-norm2','minLapse','maxAbsKret'],['Stored C-norm2','Global minimum lapse','Global maximum |Kretschmann|']):
  ax.plot(t,[x[key] for x in h],color=colors[n],lw=1.2,label=label);ax.set(xlabel='Coordinate time',ylabel=ylabel,yscale='log',title=method);ax.grid(alpha=.2);ax.legend(fontsize=8)
 for ax,key in zip(axes,['nmb_total','maxRefLev']):ax.plot(t,[x[key] for x in h],color=colors[n],ls='-' if method=='chiTE' else '--',label=f'{method} N{n}')
 summary.append(dict(method=method,N=n,directory=str(d),run_status=rc,final=h[-1],historical_min_lapse=min(x['minLapse'] for x in h),max_C2=max(x['C-norm2'] for x in h),max_blocks=max(x['nmb_total'] for x in h)))
fig.suptitle('A=-0.04896875 | dchi versus derivative-error AMR (partial histories until all runs finish)');fig.savefig(root/'comparison.png',dpi=160);fig.savefig(root/'comparison.pdf')
for ax,title in zip(axes,['Meshblock count','Maximum physical refinement level']):ax.set(xlabel='Coordinate time',ylabel=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig2.savefig(root/'mesh_comparison.png',dpi=160);(root/'series-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps([dict(method=x['method'],N=x['N'],status=x['run_status'],time=x['final']['time'],blocks=x['final']['nmb_total']) for x in summary],indent=2))
