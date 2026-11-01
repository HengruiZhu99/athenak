from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
ns=[12,18,24,32];data={o:{n:np.genfromtxt(p/f'ko{o}-n{n}.csv',delimiter=',',names=True) for n in ns} for o in [8,10]}
summary={}
for order in [8,10]:
 summary[order]={}
 for field in ['logical4','logical5','smooth','x4']:
  summary[order][field]={}
  for region in range(4):
   values=[float(data[order][n][field][region]) for n in ns]
   rates=[float(np.log(a/b)/np.log(m/n)) if min(a,b)>0 else None for a,b,n,m in zip(values,values[1:],ns,ns[1:])]
   summary[order][field][region]=dict(maxima=values,effective_orders=rates)
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for ax,region,title in zip(axes,[0,2],['Cartesian core','Outer shells, r < 0.8']):
 for o,color in [(8,'#b55315'),(10,'#1769aa')]:
  values=summary[o]['smooth'][region]['maxima'];ax.loglog(ns,values,'o-',color=color,label=f'KO{o}')
 ref=summary[10]['smooth'][region]['maxima'][-1]
 ax.loglog(ns,[ref*(32/n)**4 for n in ns],'k:',label='fourth-order reference')
 ax.set(xlabel='N',ylabel='max |Q sin(2.3x + 0.7y − 0.4z)|',title=title);ax.grid(alpha=.2);ax.legend()
fig.suptitle('Actual composite-grid dissipation consistency (all block edges included)')
fig.tight_layout(rect=(0,.06,1,.94));fig.text(.5,.01,'Same geometry, overlap, taper and coefficient. Outer-shell effective orders are not yet asymptotic.',ha='center',fontsize=9)
fig.savefig(p/'consistency.png',dpi=150)
(p/'summary.json').write_text(json.dumps(dict(resolutions=ns,regions=['Cartesian core','inner spherical shell/overlap','outer shells r<.8','r>=.8'],results=summary),indent=2)+'\n')
for o in [8,10]: print(o,'core',summary[o]['smooth'][0]['effective_orders'],'outer',summary[o]['smooth'][2]['effective_orders'])
