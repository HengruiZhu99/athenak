#!/usr/bin/env python3
"""Regenerate publication figures from the bundled, immutable data export."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parent;D=HERE/'data';F=HERE/'figures';F.mkdir(exist_ok=True)
z=json.loads((D/'evidence.json').read_text())
C=['#0072B2','#D55E00','#009E73','#8A5194','#666666']
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
 'font.size':9,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=C),
 'lines.linewidth':1.45,'savefig.dpi':200,'pdf.fonttype':42,'ps.fonttype':42})
def panels(nr=1,nc=2,height=2.55):
    fig,ax=plt.subplots(nr,nc,figsize=(7.05,height),layout='constrained',squeeze=False)
    for i,a in enumerate(ax.flat):
        a.grid(alpha=.18,which='major');a.text(.01,1.035,f'({chr(97+i)})',transform=a.transAxes,fontweight='bold')
    return fig,ax.flatten()
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)
def series(key,field):
    rows=z[key]['history'];return np.array([r['t'] for r in rows]),np.array([np.nan if r.get(field) is None else r[field] for r in rows])
def positive(x,y):
    y=np.array(y,dtype=float);return np.asarray(x)[y>0],y[y>0]

# Schematic: a method illustration, not simulated physical data.
fig,ax=plt.subplots(figsize=(7.05,2.12));ax.set(xlim=(0,10),ylim=(0,3.2));ax.axis('off')
boxes=[(.1,1.75,2.8,.95,'Independent stage fields',r'$\chi,g,A_{rr},\hat K,\Theta,\Lambda^r,\alpha$'),
 (3.6,1.75,2.8,.95,'Metric-derived shift',r'$\partial_t V^r=-\eta_D V^r$; outer nullness'),
 (7.1,1.75,2.8,.95,'Nonlinear RHS + matter',r'$\partial_t q=R_q+S_q/\Omega$'),
 (3.6,.05,2.8,.95,'Accepted RK4 update','Constraints measured, never reset')]
for i,(x,y,w,h,title,subtitle) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor='#F2F6FA',edgecolor=C[0],lw=.9))
    ax.text(x+w/2,y+.64,title,ha='center',fontsize=9.5,fontweight='bold');ax.text(x+w/2,y+.25,subtitle,ha='center',fontsize=8.5)
for p,q in [((2.95,2.23),(3.5,2.23)),((6.45,2.23),(7,2.23)),((8.5,1.68),(6.45,.53)),((3.5,.53),(1.5,1.68))]:
    ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',color='#333333',lw=1.2))
ax.text(5,3.03,'Shift solved at every Runge--Kutta stage',ha='center',fontsize=10)
save(fig,'scheme')

fig,axs=panels()
for ax,region,title in zip(axs,['bulk','endpoint'],['Bulk: $r<0.8$','Near scri: $r>0.9$']):
    g=z['gauge'];t=np.array(g['times']);a,b=g['regions'][region]['sampled_differences'];p=g['regions'][region]['maximum_order']
    ax.semilogy(*positive(t,a),'o-',ms=3,label=r'$2048\leftrightarrow4096$')
    ax.semilogy(*positive(t,b),'s-',ms=3,label=r'$4096\leftrightarrow8192$')
    ax.set(xlabel='$t$',ylabel='Physical-field difference',title=title,ylim=(1e-16,2e-9),xlim=(.05,1.02))
    ax.text(.96,.94,f'Peak order {p:.2f}',transform=ax.transAxes,ha='right',va='top')
axs[0].legend(loc='lower left');save(fig,'gauge_convergence')

fig,axs=panels()
for ax,field in zip(axs,['H','M']):
    ns=np.array([2048,4096,8192]);sub=[run['all_stage_extrema']['max_accepted_'+field+'_rms'] for run in z['subtracted']['runs']]
    kah=z['arithmetic']['constraints'][field+'_rms']['maxima']
    ax.loglog(ns,sub,'o-',label='Local subtraction');ax.loglog(ns,kah,'s-',label='+ compensated updates')
    ax.loglog(ns,kah[0]*(ns/ns[0])**-5,':',color=C[4],label='$N^{-5}$ guide')
    ax.set(xlabel='Grid points $N$',ylabel='Peak accepted-state RMS',title='Hamiltonian' if field=='H' else 'Momentum')
    ax.set_xticks(ns,[str(n) for n in ns]);ax.minorticks_off()
axs[0].legend();save(fig,'arithmetic_floor')

fig,axs=panels()
for method,bias,label in [('discrete-ode','centered','Centered shift ODE'),('discrete-ode','right','Right-biased shift ODE'),('quadrature','centered','Quadrature')]:
    rr=[r for r in z['modes']['records'] if r['method']==method and r['bias']==bias and r['epsilon']==1e-6]
    axs[0].plot([r['n'] for r in rr],[r['dominant_real'] for r in rr],'o-',ms=3,label=label)
axs[0].axhline(0,color='k',lw=.6);axs[0].set(xlabel='Grid points $N$',ylabel=r'$\max\,\mathrm{Re}\,\lambda$',yscale='symlog',title='Coupled finite-grid spectrum');axs[0].legend()
g=z['gauge_old'];t=np.array(g['times']);p=np.array([np.nan if x is None else x for x in g['regions']['endpoint']['sampled_orders']]);mask=(t>=.4)&(t<=1)
axs[1].plot(t[mask],p[mask],'o-',label='Original grids: 512 / 1024 / 2048');axs[1].axhline(3,color=C[4],ls='--',label='Declared endpoint threshold')
axs[1].set(xlabel='$t$',ylabel='Observed endpoint order',title='A final-time test misses a transient',ylim=(0,7));axs[1].legend(fontsize=7,loc='lower right')
save(fig,'endpoint_failures')

fig,axs=panels(2,2,4.4)
for n,color in zip([513,1025,2049],C):
    q=np.load(D/f'front{n}.npz');r=q['r'];mask=(r>.105)&(r<.135);d=q['diagnostic'];phys=q['q']
    axs[0].plot(r[mask],(d[:,7]/((1-r*r)/2))[mask],color=color,label=f'$N={n}$')
    axs[1].plot(r[mask],(phys[:,3]-3)[mask],color=color)
    axs[2].plot(r[mask],d[mask,0],color=color)
rr=[row for row in z['front']['records'] if row['time']==2.5]
axs[3].plot([r['n'] for r in rr],[r['maximum_physical_lapse_gradient'] for r in rr],'o-',color=C[3])
for ax,title,ylab in zip(axs[:3],['Physical lapse','Trace variable','Hamiltonian residual'],[r'$\widetilde\alpha$',r'$\hat K$',r'$\mathcal{H}$']):ax.set(xlabel='$r$',ylabel=ylab,title=title)
axs[0].legend();axs[3].set(xlabel='Grid points $N$',ylabel=r'$\max|\partial_r\widetilde\alpha|$',title='Steepening with refinement')
save(fig,'wormhole_front')

fig,axs=panels(2,2,4.4)
ns=np.array([1009,2017,4033])
for n,color in zip(ns,C):
    t,H=series(f'domain{n}','H_rms');_,M=series(f'domain{n}','M_rms')
    axs[0].semilogy(*positive(t,H),color=color,label=f'$N={n}$');axs[1].semilogy(*positive(t,M),color=color)
for ax,title in zip(axs[:2],['Hamiltonian: retained domain','Momentum: retained domain']):ax.set(xlabel='$t$',ylabel='Sampled RMS constraint',title=title,xlim=(3.2,10))
axs[0].legend()
for field,marker in [('H','o'),('M','s')]:
    vals=[z[f'domain{n}']['stages']['max_accepted_'+field+'_rms'] for n in ns]
    axs[2].loglog(ns,vals,marker+'-',label=field)
axs[2].set(xlabel='Retained grid points $N$',ylabel='Peak accepted-state RMS',title='Transient error remains unresolved')
axs[2].set_xticks(ns,[str(n) for n in ns]);axs[2].minorticks_off();axs[2].legend()
for name,label in [('compact1009','Compact metric profile'),('global1009','Global metric profile')]:
    q=np.load(D/(name+'.npz'));r=q['r'];mask=(r>.07)&(r<.15)
    axs[3].plot(r[mask],q['diagnostic'][mask,1],label=label)
axs[3].set(xlabel='$r$',ylabel=r'$\mathcal{M}_r$',title='$t=10$: a variable-profile artifact');axs[3].legend(fontsize=7)
save(fig,'retained_domain')

fig,axs=panels(2,2,4.4)
fixed=z['fixed_scalar']['spatial_runs'];ns=np.array([r['n'] for r in fixed]);er=[r['max_weighted_field_L2'] for r in fixed]
axs[0].loglog(ns,er,'o-',label='Field error');axs[0].loglog(ns,[r['max_scri_error'] for r in fixed],'s-',label='Scri error')
axs[0].loglog(ns,er[0]*(ns/ns[0])**-2,':',color=C[4],label='$N^{-2}$ guide');axs[0].legend()
axs[0].set(title='Independent fixed-background solver',xlabel='Grid points $N$',ylabel='Maximum analytic-solution error');axs[0].set_xticks(ns,[str(n) for n in ns]);axs[0].minorticks_off()
rr=sorted(z['scalar']['records'],key=lambda r:r['arguments']['n']);ns=np.array([r['arguments']['n'] for r in rr]);errs=np.abs([r['mass_minus_flux'] for r in rr])
axs[1].loglog(ns,errs,'o-',label=r'$|M_0-E_{\rm rad}|$');axs[1].loglog(ns,errs[0]*(ns/ns[0])**-5,':',color=C[4],label='$N^{-5}$ guide')
axs[1].set(title='Coupled dispersal: $A_\phi=0.058$',xlabel='Grid points $N$',ylabel='Final mass--flux discrepancy');axs[1].set_xticks(ns,[str(n) for n in ns]);axs[1].minorticks_off();axs[1].legend()
for n,color in zip([512,1024,2048],C):
    axs[2].semilogy(*positive(*series(f'dispersal{n}','H_rms')),color=color,label=f'$N={n}$')
    axs[3].semilogy(*positive(*series(f'collapse{n}','H_rms')),color=color,label=f'$N={n}$')
axs[2].set(title='Dispersal constraints',xlabel='$t$',ylabel=r'Sampled $\|\mathcal{H}\|_r$');axs[2].legend()
axs[3].set(title='Failed collapse trial: $A_\phi=0.07$',xlabel='$t$',ylabel=r'Sampled $\|\mathcal{H}\|_r$');axs[3].legend()
save(fig,'scalar_results')

fig,axs=panels()
for n,color in zip([256,512,1024],C):
    t,y=series(f'cmc{n}','H_rms');axs[0].semilogy(*positive(t,y),color=color,label=f'$N={n}$')
    t,y=series(f'cmc{n}','apparent_horizon_mass');axs[1].semilogy(*positive(t,abs(y-1/3)/(1/3)),color=color)
axs[0].set(title='Already-trumpet CMC gauge control',xlabel='$t$',ylabel=r'Sampled $\|\mathcal{H}\|_r$');axs[0].legend()
axs[1].set(title='Apparent-horizon mass',xlabel='$t$',ylabel=r'$|M_{\rm AH}-M|/M$')
save(fig,'cmc_control')
print('Wrote eight vector-PDF figures and PNG previews.')
