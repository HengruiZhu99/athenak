"""Publication figures from the archived linearized-step evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,3.8));fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.88,wspace=.28)
for eta,label,color in [(0,'Direct control','tab:orange'),(1,'Adaptive reference','tab:green')]:
 n=[];rho=[];growth=[]
 for size,name in [(25,'boundary-step-v1'),(33,'boundary-step-n33-v1'),(65,'boundary-step-n65-v1')]:
  s=json.loads((a.evidence/(name+'.json')).read_text());r=next(r for r in s['rows'] if r['eta']==eta and r['epsilon']==5e-7)
  n.append(size);rho.append(r['spectral_radius']);growth.append(np.log(r['spectral_radius'])/r['dt'])
 axes[0].plot(n,rho,'o-',label=label,color=color);axes[1].plot(n,growth,'o-',label=label,color=color)
axes[0].axhline(1,color='.5',linestyle='--',linewidth=1);axes[0].plot([33],[1],'*',color='tab:blue',markersize=11,label='Advective reference')
axes[0].set(title='Linearized RK-step amplification',xlabel='Grid nodes N',ylabel='Spectral radius')
axes[1].set(title='Growth rate of the dominant mode',xlabel='Grid nodes N',ylabel='log(spectral radius) / timestep')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(a.output/('linear-growth.'+ext),dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.7));fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.87,wspace=.3)
for eta,label,color in [(0,'Direct control','tab:orange'),(1,'Adaptive reference','tab:green')]:
 with np.load(a.evidence/f'boundary-step-n33-v1-eta{eta}-eps5e-07.npz') as z:
  v=abs(z['dominant_mode']);r=z['r'];norm=np.linalg.norm(v[:,:10]);axes[0].semilogy(r,np.maximum(v[:,2]/norm,1e-14),'-o',ms=2,label=label,color=color);axes[1].semilogy(r,np.maximum(v[:,5]/norm,1e-14),'-o',ms=2,label=label,color=color)
axes[0].set(title='Dominant mode: radial curvature',ylabel='Magnitude / physical-state mode norm')
axes[1].set(title='Dominant mode: independent connection')
for ax in axes:ax.set_xlabel('Compact radius r');ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(a.output/('boundary-mode.'+ext),dpi=180)
plt.close(fig)
