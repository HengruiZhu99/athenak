from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
coords=np.genfromtxt(ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv',delimiter=',',names=True)
r=coords['r'];weight=coords['volume'];n=len(r)
fig,axes=plt.subplots(2,3,figsize=(12,7));records={}
for label,folder,campaign in [('DO1','direct-outer-v1','n16-direct-outer-fresh10m-v1-segment01'),('IT1','inner-telegrapher-v1','n16-inner-telegrapher-fresh10m-v1-segment01')]:
 cp=ROOT/'output/hyperbolic-core-shell'/folder/'aurora/campaigns'/campaign/'output/puncture.checkpoint'
 a=np.memmap(cp,dtype='<f8',mode='r',offset=168,shape=(n,31));o=(1-r*r)/2;A=(1+r*r)/2
 alpha=a[:,18]+A;beta=a[:,22:25]-np.column_stack([coords[k] for k in ['x','y','z']]);br=(beta*np.column_stack([coords[k] for k in ['x','y','z']])).sum(axis=1)/r
 fields=[(a[:,0],'conformal chi'),(alpha,'conformal lapse'),(br,'radial shift'),(a[:,13]-3,'K - 2Theta'),(a[:,14],'Theta'),(np.sqrt((a[:,19:22]**2).sum(axis=1)),'B magnitude')]
 bins=np.linspace(0,1,51);mid=(bins[:-1]+bins[1:])/2
 record={}
 for ax,(v,title) in zip(axes.flat,fields):
  avg=np.array([np.average(v[m],weights=weight[m]) if np.sum(weight[m])>0 else np.nan for lo,hi in zip(bins[:-1],bins[1:]) for m in [(r>=lo)&(r<hi)&(weight>0)]])
  ax.plot(mid,avg,label=label);ax.set(xlabel='compactified radius',ylabel=title);ax.grid(alpha=.2);ax.axvspan(.2,.9,color='k',alpha=.035);record[title]=avg.tolist()
 records[label]=record
axes[0,0].legend();fig.suptitle('N16 at10M: volume-weighted radial profiles');fig.tight_layout();fig.savefig(HERE/'profiles-10m.png',dpi=150)
(HERE/'profiles-10m.json').write_text(json.dumps(dict(radius=mid.tolist(),profiles=records),indent=2)+'\n')
