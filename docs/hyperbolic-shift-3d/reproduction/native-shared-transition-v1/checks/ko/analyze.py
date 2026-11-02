from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
ns={o:sorted(int(q.stem.split('-n')[1]) for q in p.glob(f'ko{o}-n*.csv')) for o in [8,10]}
data={o:{n:np.genfromtxt(p/f'ko{o}-n{n}.csv',delimiter=',',names=True) for n in ns[o]} for o in [8,10]}
def orders(values,n):
 return [float(np.log(a/b)/np.log(j/i)) if min(a,b)>0 else None for a,b,i,j in zip(values,values[1:],n,n[1:])]
summary={}
for order in [8,10]:
 summary[order]={}
 for field in ['logical4','logical5','smooth','x4']:
  summary[order][field]={}
  for region in range(4):
   values=[float(data[order][n][field][region]) for n in ns[order]]
   summary[order][field][region]=dict(maxima=values,effective_orders=orders(values,ns[order]))
 for n in ns[order]:assert np.all(data[order][n]['energy']<0)
bounded=np.genfromtxt(p/'ko10-shell-bounded.csv',delimiter=',',names=True)
parity={}
for row in bounded:
 n=int(row['n'])
 if n in data[10]:
  parity[n]=float(abs(row['max_abs']/data[10][n]['smooth'][2]-1))
  assert parity[n]<1e-6,(n,parity[n])
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,region,title in zip(axes,[0,2],['Cartesian core (full grids)','Outer shells: full grids and bounded finer samples']):
 for o,color in [(8,'#b55315'),(10,'#1769aa')]:
  values=summary[o]['smooth'][region]['maxima'];ax.loglog(ns[o],values,'o-',color=color,label=f'KO{o}: full grid')
 if region==2:ax.loglog(bounded['n'],bounded['max_abs'],'x--',color='#228844',label='KO10: bounded face/edge/corner samples')
 ref=summary[10]['smooth'][region]['maxima'][-1];grid=ns[10] if region==0 else bounded['n']
 ax.loglog(grid,[ref*(max(ns[10])/n)**4 for n in grid],'k:',label='Fourth-order reference')
 ax.set(xlabel='N',ylabel='max |Q sin(2.3x + 0.7y − 0.4z)|',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Dissipation consistency: separate KO10 repair with unchanged geometry and taper')
fig.tight_layout(rect=(0,.08,1,.94));fig.text(.5,.02,'Full tests include all block edges. Bounded tests cover all six faces of both non-overlap shells; they are not finer full-grid evolutions.',ha='center',fontsize=9)
fig.savefig(p/'consistency.png',dpi=150)
r=dict(resolutions=ns,regions=['Cartesian core','inner spherical shell/overlap','outer shells r<.8','r>=.8'],results=summary,
 bounded_shell=dict(resolutions=bounded['n'].tolist(),maxima=bounded['max_abs'].tolist(),effective_orders=orders(bounded['max_abs'],bounded['n']),scaled_n4=bounded['scaled_n4'].tolist(),full_grid_max_relative_difference=parity))
(p/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
print('full-grid KO10 orders',summary[10]['smooth'][2]['effective_orders']);print('bounded shell',r['bounded_shell'])
