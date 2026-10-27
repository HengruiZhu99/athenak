#!/usr/bin/env python3
"""Render raw inner lapse profiles and the reproduced rejected RK stages."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();z=np.load(a.data/'lapse-profiles.npz');st=np.load(a.data/'stages.npz')
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
f,axs=plt.subplots(1,2,figsize=(8.3,3.5),layout='constrained')
for t,c in zip([10,20,30,40,43.8],plt.cm.viridis(np.linspace(.05,.9,5))):
 r=z[f't{round(t*100)}_r'];v=z[f't{round(t*100)}_physical'];mask=(r>0)&(r<=1e-5);axs[0].loglog(r[mask],v[mask],color=c,label=f'$t={t:g}$',lw=1.4)
axs[0].set(xlabel='$r$',ylabel=r'Physical lapse $\widetilde\alpha$',title='Inner lapse becomes oscillatory');axs[0].legend(ncol=2,fontsize=8)
r=st['r'];O=(1-r*r)/2;ref=(1+r*r)/2*r*r/(r*r+O**4)
for stage,c in zip([1,2,3],['#0072B2','#D55E00','#009E73']):
 alpha=ref[1:13]*np.exp(st[f'q_stage{stage}'][1:13,6])/O[1:13];axs[1].semilogy(np.arange(1,13),alpha,'o-',ms=3,color=c,label=f'RK stage {stage}',lw=1.3)
axs[1].set(xlabel='Positive-radius node index',ylabel=r'Physical lapse $\widetilde\alpha$',title='Rejected step at $t=43.8365$',xticks=[1,2,4,6,8,10,12]);axs[1].legend(loc='upper left',fontsize=8)
axs[1].text(.44,.12,r'Stage 4: node 2 underflows to zero'+'\n'+r'$\log\alpha=-1109.3$'+'\n'+r'$r_2=3.63\times10^{-9}$',transform=axs[1].transAxes,fontsize=8,color='#A02020',bbox=dict(facecolor='white',edgecolor='.8',alpha=.95))
for ax in axs:ax.grid(alpha=.18)
for ext in ['pdf','png']:f.savefig(a.data/('lapse-stop.'+ext),dpi=210,bbox_inches='tight')
plt.close(f)
