#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'docs/z4c_hyperboloidal/evidence/scri-investigation/symbol'
data=json.loads((root/'symbols.json').read_text());exact=json.loads((root/'radial-limit.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for angle,label in [(0,'Radial'),(.35,'20 degrees'),(np.pi/4,'45 degrees'),(1.2,'69 degrees'),(np.pi/2,'Tangential')]:
 rows=[v for v in data if v['background']=='wormhole' and v['collar_begin']==.98 and v['radius']==.9999 and abs(v['theta']-angle)<1e-10]
 ax[0].plot([v['epsilon'] for v in rows],[v['max_imag'] for v in rows],marker='.',label=label)
ax[0].axvline(exact['epsilon_critical'],color='.4',ls=':',label='Radial limiting threshold')
ax[0].set(xlabel='Source multiplier epsilon',ylabel='Maximum imaginary characteristic part',title='Full coupled symbol near scri');ax[0].legend(frameon=False,fontsize=9)
eps=np.linspace(0,1,401);roots=np.array([np.roots([1,0,-3,13*e/6,2])-1 for e in eps])
ax[1].plot(eps,np.max(abs(roots.imag),axis=1),color='#b54638',label='Exact radial scri limit')
fourier=json.loads((root/'fourier-check.json').read_text());rows=[v for v in fourier if v['background']=='wormhole' and v['epsilon']==1]
ax[1].scatter([1]*len(rows),[v['max_growth_per_wavenumber'] for v in rows],color='#17699c',marker='x',label='Full Fourier growth/k, increasing k')
ax[1].set(xlabel='Source multiplier epsilon',ylabel='Growth rate / wavenumber',title='Independent high-frequency check');ax[1].legend(frameon=False,fontsize=9)
for a in ax:a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
fig.suptitle('Full-strength proposal C fails the hyperbolicity gate',fontsize=15)
for ext in ['png','pdf']:fig.savefig(root/f'C-characteristic-failure.{ext}',dpi=180)
