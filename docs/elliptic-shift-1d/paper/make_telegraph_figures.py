#!/usr/bin/env python3
"""Redraw follow-up figures from the portable, verified telegrapher export."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'data/telegraph';F=P/'figures'
z=json.loads((D/'summary.json').read_text());ray=json.loads((D/'rays.json').read_text());q=np.load(D/'profiles.npz',allow_pickle=False)
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':9,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200,'pdf.fonttype':42})
C={1025:'#D55E00',2049:'#0072B2',4097:'#009E73'};styles={1025:'--',2049:'-',4097:':'}
def panels(nr=1,nc=2,height=2.65,ratios=None):
 fig,axs=plt.subplots(nr,nc,figsize=(7.05,height),layout='constrained',squeeze=False,gridspec_kw={'height_ratios':ratios} if ratios else None)
 for i,ax in enumerate(axs.flat):ax.grid(alpha=.18);ax.text(.01,1.035,f'({chr(97+i)})',transform=ax.transAxes,fontweight='bold')
 return fig,axs

def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),bbox_inches='tight')
 plt.close(fig)
def plot_history(ax,key):
 for rec in z['records']:
  n=rec['n'];t=[r['t'] for r in rec['rows']];y=np.array([r[key] for r in rec['rows']]);ax.plot(t,np.ma.masked_less_equal(y,0),color=C[n],ls=styles[n],label=f'$N={n}$')
 ax.set(xlabel='$t$',xlim=(0,10),yscale='log')

def profile(n,t):
 key=f'n{n}_t{round(t*100):04d}';return {k:q[key+'_'+k] for k in ['r','alpha','physical','H']}
fig,axs=panels()
times=[0,1,2.5,4,6,8,10];colors=plt.cm.viridis(np.linspace(.05,.9,len(times)))
for t,c in zip(times,colors):
 d=profile(4097,t);axs[0,0].plot(d['r'],d['alpha'],color=c,label=f'$t={t:g}$')
 if t>=4:
  m=(d['r']>=1e-8)&(d['r']<=.03);axs[0,1].semilogx(d['r'][m],d['physical'][m],color=c,label=f'$t={t:g}$')
axs[0,0].set(xlabel='$r$',ylabel=r'$\alpha$',title='Global conformal lapse',xlim=(0,1));axs[0,0].legend(ncol=2,fontsize=7)
axs[0,1].set(xlabel='$r$',ylabel=r'$\widetilde\alpha$',title='Physical lapse near the puncture',xlim=(1e-8,.03));axs[0,1].legend(fontsize=7)
save(fig,'telegraph_lapse_evolution')
fig,axs=panels(2,2,4.3,ratios=[2.5,1])
for col,(t,lo,hi) in enumerate([(2.5,.03,.22),(10.,1e-8,3e-5)]):
 for n in [1025,2049,4097]:
  d=profile(n,t);m=(d['r']>=lo)&(d['r']<=hi);axs[0,col].plot(d['r'][m],d['physical'][m],color=C[n],ls=styles[n],label=f'$N={n}$')
 for na,nb in [(1025,2049),(2049,4097)]:
  a,b=profile(na,t),profile(nb,t);m=(a['r']>=lo)&(a['r']<=hi);np.testing.assert_array_equal(a['r'],b['r'][::2]);axs[1,col].plot(a['r'][m],a['physical'][m]-b['physical'][::2][m],color=C[na],label=f'{na} - {nb}')
 axs[0,col].set(title=f'Physical lapse at $t={t:g}$',ylabel=r'$\widetilde\alpha$');axs[0,col].legend(fontsize=7);axs[0,col].tick_params(labelbottom=False)
 axs[1,col].set(xlabel='$r$',ylabel=r'$\Delta\widetilde\alpha$');axs[1,col].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axs[1,col].legend(fontsize=6)
 axs[1,col].texts[0].set_position((.02,.83));axs[1,col].texts[0].set_bbox(dict(facecolor='white',edgecolor='none',pad=1))
 for ax in axs[:,col]:ax.set_xlim(lo,hi)
 if col:
  for ax in axs[:,col]:ax.set_xscale('log')
save(fig,'telegraph_lapse_resolution')
fig,axs=panels(2,2,4.5)
for ax,key,title,symbol in zip(axs.flat,['H','M','Theta','Z'],['Hamiltonian','Radial momentum','Z4 scalar','Connection constraint'],[r'\mathcal{H}',r'\mathcal{M}_r',r'\Theta',r'Z_r']):
 plot_history(ax,key+'_rms');ax.set(title=title,ylabel=rf'$\|{symbol}\|_r$')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False);save(fig,'telegraph_constraints')
fig,axs=panels(2,2,4.5);tt=q['ray_times'];rr=q['rays'];active_time=tt>=1
for seed,c in zip(range(0,81,10),plt.cm.viridis(np.linspace(.05,.9,9))):
 for f in range(2):
  radius=rr[:,2*seed+f,0];active=active_time&(radius>0)&(radius<1);axs[0,f].plot(tt,np.where(active,radius,np.nan),color=c,lw=.9)
axs[0,0].set(title='Inward characteristic family',ylabel='$r$',yscale='log');axs[0,1].set(title='Outward family; terminal exits',ylabel='$r$',ylim=(0,1.02))
for rec in ray['histories']:
 n=rec['n'];rows=[r for r in rec['rows'] if r['t']>=1]
 axs[1,0].plot([r['t'] for r in rows],[r['families'][0]['min_active_logJ'] for r in rows],color=C[n],ls=styles[n],label=f'$N={n}$')
for pair in ray['pairs']:
 t=[r['t'] for r in pair['history']];y=np.array([r['logJ'] for r in pair['history']]);axs[1,1].semilogy(t,np.ma.masked_less_equal(y,0),label=f"{pair['grids'][0]} - {pair['grids'][1]}")
axs[1,0].set(title='Inward compression',ylabel=r'$\min\log\mathcal{J}_-$');axs[1,0].legend(fontsize=7)
axs[1,1].set(title='All-ray spatial differences',ylabel=r'$\max|\Delta\log\mathcal{J}|$');axs[1,1].legend(fontsize=7)
for ax in axs.flat:ax.set(xlabel='$t$',xlim=(1,10))
axs[0,1].set_xlim(1,2.5)
save(fig,'telegraph_characteristics')
fig,axs=panels(2,2,4.5)
plot_history(axs[0,0],'H_max');axs[0,0].set(title='Full-grid Hamiltonian maximum',ylabel=r'$\max_r|\mathcal{H}|$');axs[0,0].legend(fontsize=7)
plot_history(axs[0,1],'lapse_gradient');axs[0,1].set(title='Physical lapse gradient',ylabel=r'$\max_{r<0.3}|\partial_r\widetilde\alpha|$')
fine=z['records'][-1];rows=fine['rows'];t=[r['t'] for r in rows]
axs[1,0].semilogy(t,[r['H_peak_r'] for r in rows],label='H peak');axs[1,0].semilogy(t,[r['AH_r'] for r in rows],'--',label='Apparent horizon');axs[1,0].set(xlabel='$t$',ylabel='$r$',title='Error localization, finest grid');axs[1,0].legend(fontsize=7)
plot_history(axs[1,1],'AH_mass_error');axs[1,1].set(title='Apparent-horizon mass',ylabel=r'$|M_{\rm AH}-1/3|$')
save(fig,'telegraph_inner_errors')
print('Five telegrapher figures regenerated from portable data.')
