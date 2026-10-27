#!/usr/bin/env python3
"""Plot only portable saved results from the single-resolution duration pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--full-domain-control',action='store_true');options=parser.parse_args()
P=Path(__file__).resolve().parent;D=P/'data/long-puncture'
s=json.loads((D/'summary.json').read_text());z=np.load(D/'profiles.npz',allow_pickle=False)
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':9,'axes.labelsize':10,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.05,4.8),layout='constrained')
rows=s['rows'];t=np.array([r['t'] for r in rows]);end=t[-1]
def vals(k):return np.array([r[k] if r[k] is not None else np.nan for r in rows])
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
for k,label,c in zip(['H','M','Theta','Z'],[r'$\mathcal{H}$',r'$\mathcal{M}_r$',r'$\Theta$',r'$Z_r$'],colors):axs[0,0].semilogy(t,np.ma.masked_less_equal(vals(k+('_rms' if options.full_domain_control else '_exterior_rms')),0),color=c,label=label)
axs[0,0].set(xlabel='$t$',ylabel='Radial RMS',title='Full-domain constraints' if options.full_domain_control else 'Exterior constraint budget');axs[0,0].legend(ncol=2)
for k,label,c in zip(['H_max','H_outside_AH'],['Full grid','Outside apparent horizon'],colors):axs[0,1].semilogy(t,np.ma.masked_less_equal(vals(k),0),label=label,color=('.55' if k=='H_max' and not options.full_domain_control else c),ls=('--' if k=='H_max' and not options.full_domain_control else '-'),lw=(1 if k=='H_max' and not options.full_domain_control else 1.6))
axs[0,1].set(xlabel='$t$',ylabel=r'$\max|\mathcal{H}|$',title='Pointwise Hamiltonian error');axs[0,1].legend()
chosen=sorted(set([v for v in s['profile_times'] if v in [6,8,10,12,14,20,50,100,250,500]]+[end]))
if len(chosen)>7:chosen=[chosen[i] for i in np.linspace(0,len(chosen)-1,7,dtype=int)]
for v,c in zip(chosen,plt.cm.viridis(np.linspace(.05,.9,len(chosen)))):
 key=f't{round(v*100):06d}';r=z[key+'_r'];a=z[key+'_physical'];mask=(r>0)&(r<.03)
 axs[1,0].semilogx(r[mask],a[mask],color=c,label=f'$t={v:g}$')
axs[1,0].set(xlabel='$r$',ylabel=r'$\widetilde\alpha$',title='Physical lapse near the puncture');axs[1,0].legend(ncol=2)
for k,label,c in (zip(['AH_mass_error','AF_mass_error','MS_mass_error'],['Apparent-horizon mass','AF-end mass','Finite-node mass maximum'],colors) if options.full_domain_control else [('AH_mass_error','Apparent-horizon mass',colors[0])]):axs[1,1].semilogy(t,np.ma.masked_less_equal(vals(k)*(1 if options.full_domain_control else 3),0),color=c,label=label)
axs[1,1].set(xlabel='$t$',ylabel=(r'$|m-M|$' if options.full_domain_control else r'$|M_{\rm AH}-M|/M$'),title='Invariant mass diagnostics' if options.full_domain_control else 'Primary invariant: horizon mass');axs[1,1].legend()
for i,ax in enumerate(axs.flat):
 ax.grid(alpha=.18);ax.text(.01,1.035,f'({chr(97+i)})',va='bottom',transform=ax.transAxes,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=1))
 if i!=2:ax.set_xlim(0,end);ax.axvline(10,color='.5',ls=':',lw=.7,zorder=0)
fig.suptitle(f'$N={s["resolution"]}$ puncture pilot: target $t=500$, evidence through $t={end:g}$',fontsize=11)
for ext in ['pdf','png']:fig.savefig(P/'figures'/(('telegraph_long_puncture_full_domain' if options.full_domain_control else 'telegraph_long_puncture')+'.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)
