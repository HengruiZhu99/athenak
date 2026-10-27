#!/usr/bin/env python3
"""Portable terminal diagnostics for the separate floored puncture experiment."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'data/floored-long-puncture'
s=json.loads((D/'summary.json').read_text());extra=json.loads((D/'floor-history.json').read_text())
rows=s['rows'];t=np.array([r['t'] for r in rows]);mass=s['parameters']['mass'];stop=s['integrator_time']
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(8.3,6.1),layout='constrained');a,b,c,d=axs.flat
for key,symbol,color in zip(['H','M','Theta','Z'],[r'\mathcal{H}',r'\mathcal{M}_r',r'\Theta',r'Z_r'],['#0072B2','#D55E00','#009E73','#CC79A7']):
 y=np.array([r[key+'_exterior_rms'] for r in rows]);assert np.isfinite(y).all();a.semilogy(t,np.ma.masked_less_equal(y,0),label=f'${symbol}$',color=color,lw=1.5)
a.set(title='(a) Exterior constraint norms',ylabel=r'$\|C\|_{r,\,r\geq r_{\rm AH}(t)}$');a.legend(ncol=2,loc='lower right')
y=np.array([r['AH_mass_error']/mass for r in rows]);b.semilogy(t,np.ma.masked_less_equal(y,0),color='#0072B2',lw=1.5)
b.set(title='(b) Apparent-horizon mass error',ylabel=r'$|M_{\rm AH}-M|/M$');b.annotate(f'Final saved: {y[-1]:.2e}',(t[-1],y[-1]),xytext=(-5,-28),textcoords='offset points',ha='right',fontsize=9,arrowprops={'arrowstyle':'-','color':'.4'})
with np.load(D/'lapse-selected.npz') as z:
 for time,color in zip([10.,40.,60.,80.,84.75],['#440154','#3B528B','#21918C','#5EC962','#E69F00']):
  prefix='t%06d'%round(time*100);r=z[prefix+'_r'];y=z[prefix+'_physical'];mask=(r>0)&(r<=1e-5);c.loglog(r[mask],y[mask],label=f'$t={time:g}$',color=color,lw=1.25)
c.axhline(s['parameters']['lapse_floor'],color='.3',ls='--',lw=1,label=r'Floor $10^{-10}$');c.set(title='(c) Raw inner physical lapse',xlabel='$r$',ylabel=r'$\widetilde\alpha$',xlim=(1.5e-9,1e-5));c.legend(ncol=2,fontsize=8,loc='lower right')
h=extra['history'];d.semilogy([x['t'] for x in h],[x['floor_max_log_correction'] for x in h],color='#D55E00',lw=1.5)
terminal=extra['terminal_stage_diagnostics']['max_lapse_floor_log_correction'];d.plot(stop,terminal,'x',color='#B22222',ms=6,label='Terminal diagnostic')
d.set(title='(d) Accumulated maximum floor correction',ylabel=r'$\max(\Delta\log\alpha)$');d.legend(loc='upper left',fontsize=8)
for ax in [a,b,d]:ax.set(xlabel='$t$',xlim=(0,stop*1.035));ax.axvline(stop,color='#B22222',ls='--',lw=1)
for ax in axs.flat:ax.grid(alpha=.18)
fig.suptitle(r'$N=1025$, physical lapse floor $10^{-10}$: stop at $t=84.7983$',fontsize=12)
for ext in ['pdf','png']:fig.savefig(P/'figures'/('telegraph_floored_long.'+ext),dpi=210,bbox_inches='tight')
plt.close(fig)
