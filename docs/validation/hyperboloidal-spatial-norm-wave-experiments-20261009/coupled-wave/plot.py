from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
w=Path(__file__).resolve().parent
fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
colors={'ray':'#b34b35','mls':'#286b9e'}
for c in ['ray','mls']:
 for N,ls in [(16,':'),(20,'--'),(24,'-')]:
  r=json.loads((w/f'N{N}-span2.2-{c}-ko0.1-fixed-dt0.1.json').read_text());h=r['history'];t=np.array([x['time'] for x in h]);e=np.array([x['error_rms'] for x in h]);ax[0].semilogy(t[1:],e[1:],color=colors[c],ls=ls,label=f'{c.upper()} N{N}')
 r=json.loads((w/f'N24-span2.1-{c}-ko0.1-dt0.1-receipt.json').read_text());h=json.loads(Path(r['pulse']['history_path']).read_text());t=np.array([x['time'] for x in h]);E=np.array([x['killing_energy'] for x in h]);ex=np.array([x['sampled_exact_killing_energy'] for x in h]);ax[1].semilogy(t,E/E[0],color=colors[c],label=c.upper())
 if c=='ray':ax[1].semilogy(t,ex/ex[0],color='#222',ls='--',label='exact sampled')
ax[0].set(xlabel='t (8.045 outward crossings at t=6)',ylabel='RMS error of φ and Π',title='Exact nonspherical dipole, common span 2.2',xlim=(0,6));ax[0].legend(fontsize=8,ncol=2);ax[0].grid(alpha=.2)
ax[1].set(xlabel='t',ylabel='Killing energy / initial energy',title='Production N24 geometry, span 2.1',ylim=(1e-9,2),xlim=(0,6));ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
fig.savefig(w/'coupled-wave.png',dpi=180)
