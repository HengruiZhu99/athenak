from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
fig,axs=plt.subplots(1,3,figsize=(14,4));summary={}
def smooth(y):
 y=np.clip(y,0,1);x=np.minimum(y,1-y);v=x**5*(126+x*(-420+x*(540+x*(-315+70*x))));return np.where(y>.5,1-v,v)
for label in ['IT1','DO1']:
 d=json.loads((HERE/f'{label}-10m-radial.json').read_text());rows=d['radial_cumulative'];r=np.array([x['cutoff'] for x in rows]);v=np.array([x['volume'] for x in rows]);volume=v[:-1]-v[1:]
 for ax,key in zip(axs[:2],['H2','M2']):
  z=np.array([x[key] for x in rows]);power=z[:-1]-z[1:];ax.stairs(np.sqrt(np.maximum(power,0)/volume),r,label=label)
 summary[label]=dict(common_exterior=d['norms']['exterior'],fraction_H2_r016_030=(next(x['H2'] for x in rows if x['cutoff']==.16)-next(x['H2'] for x in rows if x['cutoff']==.3))/rows[0]['H2'],fraction_H2_r090_1=next(x['H2'] for x in rows if x['cutoff']==.9)/rows[0]['H2'])
r=np.linspace(0,.999,500);o=(1-r*r)/2;alpha=(1+r*r)/2;w=smooth((.8-r)/.24)
for label,s,old in [('IT1',smooth((.56-r)/.36),4*alpha+w*(8*alpha-4*alpha)),('DO1',smooth((.9-r)/.7),4*alpha)]:
 rate=((1-s)*old*1.01+s*.06*alpha)/o
 axs[2].plot(r,rate,label=label)
for ax,title in zip(axs,['Hamiltonian RMS in radial bands at10M','Momentum magnitude RMS in radial bands at10M','Theta damping coefficient on initial CMC reference']):
 ax.set(xlabel='compactified radius',title=title,yscale='log');ax.grid(alpha=.2);ax.axvspan(.16,.18,alpha=.12,color='k',label='core/shell overlap' if ax==axs[0] else None);ax.legend(fontsize=8)
axs[2].set_ylim(.03,1e3)
fig.suptitle('N16 controlled comparison: common grid and cutoff r>0.1');fig.tight_layout();fig.savefig(HERE/'radial-and-damping.png',dpi=150)
(HERE/'radial-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
