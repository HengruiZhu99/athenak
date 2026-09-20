"""Portable figure regeneration from the compact package: python scripts/render_package.py."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
assert (root/'histories').is_dir(), 'Run from the compact results package'

def series(job,case,kind):
 p=root/'histories'/(job+'-'+case+'.npz')
 if not p.exists():return None
 with np.load(p,allow_pickle=False) as f:return {str(n):f[kind][:,i] for i,n in enumerate(f[kind+'_columns'])}
fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained',facecolor='white')
for job,case,label in [('8842172','baseline',r'no sponge, $\kappa_1=.1$'),('8842172','radial_k01',r'radial, $\kappa_1=.1$'),('8842171','radial_k0',r'radial, $\kappa_1=0$')]:
 h=series(job,case,'user');c=series(job,case,'z4c')
 if h is None:continue
 for ax,t,y in zip(axes,(h['time'],c['time'],h['time']),(h['Theta-max'],np.sqrt(c['Theta-norm']/c['Volume']),h['alpha-res'])):ax.semilogy(t,y,label=label)
for ax,label,floor in zip(axes,(r'$\max|\Theta|$',r'Whole-domain $\Theta$ RMS',r'$\max|\delta\alpha|$'),(1e-13,1e-14,1e-13)):ax.set_xlabel('t (code units)');ax.set_ylabel(label);ax.axhline(floor,color='.5',ls=':',lw=.9);ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=8);fig.savefig(root/'long-growth-comparison.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained',facecolor='white')
for job,case,label,color in [('8842248','theta_primary',r'$d_\alpha=.01$','C0'),('8842248','theta_lapse01',r'$d_\alpha=.1$','C1'),('8842283','theta_amplitude',r'$A/10, d_\alpha=.01$','C2')]:
 h=series(job,case,'user');c=series(job,case,'z4c')
 if h is None:continue
 axes[0].semilogy(h['time'],h['Theta-max'],color=color,label=label);axes[1].semilogy(c['time'],np.sqrt(c['Theta-norm']/c['Volume']),color=color);axes[2].semilogy(c['time'],np.sqrt(c['Theta-int2']),color=color,label=label+' core');axes[2].semilogy(c['time'],np.sqrt(c['Theta-int2']+c['Theta-norm']),color=color,ls='--',label=label+' whole')
for ax,label in zip(axes,(r'$\max|\Theta|$',r'Exterior $\Theta$ RMS',r'$\sqrt{\int\Theta^2dV}$')):ax.set_xlabel('t (code units)');ax.set_ylabel(label);ax.grid(alpha=.15)
axes[0].axhline(1e-13,color='.5',ls=':',lw=.9);axes[1].axhline(1e-14,color='.5',ls=':',lw=.9);axes[0].legend(frameon=False,fontsize=8);axes[2].legend(frameon=False,fontsize=7);fig.savefig(root/'theta-comparison.png',dpi=180);plt.close(fig)
profiles=json.loads((root/'profile-summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained',facecolor='white')
for key,label in [('8842171-radial_k0','Lapse pulse'),('8842248-theta_primary',r'$\Theta$ pulse, $d_\alpha=.01$'),('8842248-theta_lapse01',r'$\Theta$ pulse, $d_\alpha=.1$'),('8842283-theta_amplitude',r'$\Theta$ pulse / 10')]:
 ds=profiles.get(key,[])
 for ax,field in zip(axes,('chi_res','conformal_metric_res','beta_res')):
  if ds:ax.semilogy([d['time_code'] for d in ds],[abs(d['active_peaks'][field]['value']) for d in ds],marker='.',label=label)
for ax,label in zip(axes,(r'$\max|\delta\chi|$',r'$\max\|\delta\widetilde\gamma\|_F$',r'$\max\|\delta\beta\|_2$')):ax.set_xlabel('t (code units)');ax.set_ylabel(label);ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=8);fig.savefig(root/'geometry-drift.png',dpi=180);plt.close(fig)

# Reconstruct the fourth figure from packaged radial checkpoint profiles.
fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained',facecolor='white')
for key,label in [('8842172-baseline',r'no sponge, $\kappa_1=.1$'),('8842172-radial_k01',r'radial, $\kappa_1=.1$'),('8842171-radial_k0',r'radial, $\kappa_1=0$')]:
 candidates=[]
 for path in (root/'profiles').glob(key+'-*.json'):
  if '.files.' in path.name:continue
  d=json.loads(path.read_text())
  if 4999<d['time_code']<5002:candidates.append(d)
 if not candidates:continue
 assert len(candidates)==1
 b=candidates[0]['radial_bins'];r=(np.array(b['edges'][1:])+np.array(b['edges'][:-1]))/2
 for ax,field in zip(axes,('Theta_RMS','chi_res_RMS','conformal_metric_res_RMS')):ax.semilogy(r,b[field],label=label)
for ax,label in zip(axes,(r'Radial-bin $\Theta$ RMS',r'Radial-bin $\delta\chi$ RMS',r'Radial-bin $\delta\widetilde\gamma$ RMS')):ax.set_xlabel('r (code units)');ax.set_ylabel(label);ax.axvline(512,color='.6',ls=':');ax.axvline(1792,color='.6',ls=':');ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=8);fig.savefig(root/'radial-profile-5000.png',dpi=180);plt.close(fig)

# Physical constraints retain the same exterior proper-volume normalization.
fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained',facecolor='white')
for job,case,label in [('8842248','theta_primary','Primary: lapse damping 0.01'),('8842248','theta_lapse01','Lapse damping 0.1'),('8842283','theta_amplitude','Theta seed / 10')]:
 c=series(job,case,'z4c')
 if c is None:continue
 for ax,field in zip(axes,['H-norm2','M-norm2','Z-norm2']):ax.semilogy(c['time'],np.sqrt(c[field]/c['Volume']),label=label)
for ax,label in zip(axes,['Exterior Hamiltonian RMS','Exterior momentum RMS','Exterior Z RMS']):ax.set_xlabel('t (M)');ax.set_ylabel(label);ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=8);fig.savefig(root/'physical-constraint-comparison.png',dpi=180);plt.close(fig)
