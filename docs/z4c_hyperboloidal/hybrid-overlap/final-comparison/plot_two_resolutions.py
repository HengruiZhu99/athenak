"""Plot only the new N12/N16 constraint and scri time histories."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
manifest=json.loads((R/'input-manifest.json').read_text())
H={};S={}
for n,dt in [(12,.00046944203735617445),(16,.00035208152801713085)]:
 for kind,dest in [('z4c_hyperboloidal',H),('scri',S)]:
  rows={}
  for p in sorted((R/'histories'/f'N{n}'/'full').glob(f'segment-*/*.{kind}.hst')):
   assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[str(p.relative_to(R))]['sha256']
   d=np.loadtxt(p,ndmin=2);assert np.isfinite(d).all() and np.all(d[:,1]==dt)
   for r in d:rows[round(r[0]/dt)]=r
  dest[n]=np.array([v for c,v in sorted(rows.items())])
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
colors={12:'#167db3',16:'#b23b67'}
fig,axs=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
for j,(ax,title,scale) in enumerate(zip(axs.flat,['Hamiltonian H','Momentum M',r'$\Theta$',r'$Z$'],[1/9,1/9,1/3,1/3])):
 for n in [12,16]:ax.semilogy(H[n][:,0]*3,H[n][:,8+j]*scale,color=colors[n],label=f'N{n}',lw=2)
 ax.set(title=title,xlabel=r'$t/M_0$',ylabel=r'$M_0^2$ RMS' if j<2 else r'$M_0$ RMS',xlim=(0,30));ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Unmasked full-overlap agreement: exterior constraint norms',fontsize=15)
for ext in ['png','pdf']:fig.savefig(R/f'two-resolution-constraints.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for ax,idx,title in zip(axs.flat,[2,3,6,7,8,21],['Compatibility residual (comp-max)','Clock diagnostic (clock-max)',r'$|\chi Q-1|$',r'$|P+3|$,  $P=K-2\Theta$',r'$|\Theta|$','RHS amplitude (rhs-max)']):
 for n in [12,16]:ax.semilogy(S[n][:,0]*3,np.ma.masked_less_equal(S[n][:,idx],0),color=colors[n],label=f'N{n}',lw=2)
 ax.set(title=title,xlabel=r'$t/M_0$',ylabel='Maximum, native conformal units',xlim=(0,30));ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Unmasked full-overlap agreement: scri-boundary diagnostics\nMaxima over boundary nodes and RK stages since each previous output',fontsize=14)
for ext in ['png','pdf']:fig.savefig(R/f'two-resolution-boundary.{ext}',dpi=180)
plt.close(fig)
