#!/usr/bin/env python3
"""Render the exterior constraint budget and primary horizon-mass diagnostic."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=P/'data/long-puncture');p.add_argument('--output',type=Path,default=P/'figures/telegraph_long_exterior_status');a=p.parse_args()
s=json.loads((a.data/'summary.json').read_text());rows=s['rows'];t=np.array([r['t'] for r in rows]);mass=s['parameters']['mass']
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
f,axs=plt.subplots(1,2,figsize=(8.3,3.35),layout='constrained')
for key,symbol,color in zip(['H','M','Theta','Z'],[r'\mathcal{H}',r'\mathcal{M}_r',r'\Theta',r'Z_r'],['#0072B2','#D55E00','#009E73','#CC79A7']):
 y=np.array([r[key+'_exterior_rms'] for r in rows]);assert np.isfinite(y).all()
 axs[0].semilogy(t,np.ma.masked_less_equal(y,0),label=f'${symbol}$',color=color,lw=1.65)
axs[0].set(title='Exterior constraint norms',ylabel=r'$\|C\|_{r,\,r\geq r_{\rm AH}(t)}$');axs[0].legend(ncol=2,loc='lower right')
y=np.array([r['AH_mass_error']/mass for r in rows]);assert np.isfinite(y).all()
axs[1].semilogy(t,np.ma.masked_less_equal(y,0),color='#0072B2',lw=1.7)
axs[1].plot(t[-1],y[-1],'o',color='#0072B2',ms=4)
axs[1].annotate(f'Final saved: {y[-1]:.2e}',xy=(t[-1],y[-1]),xytext=(-8,-28),textcoords='offset points',ha='right',fontsize=9,arrowprops=dict(arrowstyle='-',color='.4'))
axs[1].set(title='Apparent-horizon mass error',ylabel=r'$|M_{\rm AH}-M|/M$')
stop=s['integrator_time'];limit=max(t[-1],stop)*1.035
for ax in axs:
 ax.set(xlabel='$t$',xlim=(0,limit));ax.grid(alpha=.2);ax.axvline(10,color='.65',ls=':',lw=.8)
 if s['status']=='failed':ax.axvline(stop,color='#B22222',ls='--',lw=1)
f.suptitle(f'$N={s["resolution"]}$: stopped at $t={stop:.4f}$; saved profiles through $t={t[-1]:g}$' if s['status']=='failed' else f'$N={s["resolution"]}$: evidence through $t={t[-1]:g}$',fontsize=12)
a.output.parent.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:f.savefig(a.output.with_suffix('.'+ext),dpi=210,bbox_inches='tight')
plt.close(f)
