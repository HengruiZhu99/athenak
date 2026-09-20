from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
p=Path(__file__).resolve().parent;j=json.loads((p/'mesh-audit.json').read_text());blocks=j['cases']['mesh16_r1']['blocks_geometry']
fig,axs=plt.subplots(1,3,figsize=(12,4),facecolor='white',layout='constrained');colors={1.:'#c7c7c7',.5:'#88bde6',.25:'#4e8cca',.125:'#1f4f82'}
for ax,limit in zip(axs[:2],[32,5]):
 for b in blocks:
  lo,hi=np.array(b['min']),np.array(b['max'])
  if lo[2]<=0<hi[2]:ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],fill=False,ec=colors[b['dx_M']],lw=.6))
 ax.add_patch(Circle((0,0),1,color='#262626',alpha=.7));ax.add_patch(Circle((0,0),4,fill=False,ls='--',ec='#262626',lw=1));ax.add_patch(Circle((2.5,0),.75,color='#e8a33c',alpha=.6))
 if limit==32:
  for r in [8,28]:ax.add_patch(Circle((0,0),r,fill=False,ls=':',ec='#a23c45',lw=1.1))
 ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel=r'$x/M$',ylabel=r'$y/M$',aspect='equal')
axs[0].set_title('232 blocks; domain ±32M',fontsize=10);axs[1].set_title('Horizon r=1M; dx=0.125M',fontsize=10)
r=np.linspace(0,32,1000);q=np.clip((r-8)/20,0,1);sigma=.05*q**3*(10+q*(-15+6*q));axs[2].plot(r,sigma,color='#a23c45');axs[2].axvspan(0,4,color='#d8e5f2');axs[2].axvspan(1.75,3.25,color='#e8a33c',alpha=.5);axs[2].axvline(8,color='0.5',ls=':');axs[2].axvline(28,color='0.5',ls=':');axs[2].set(xlabel=r'$r/M$',ylabel=r'Outer damping $sigma M$',xlim=(0,32),ylim=(0,.054));axs[2].set_title('No damping inside r=8M',fontsize=10);axs[2].grid(alpha=.2)
fig.savefig(p/'mesh-and-layer.png',dpi=180);fig.savefig(p/'mesh-and-layer.pdf')
