from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
base=P.parent/'native-shared-transition-v1/checks/initial-grid'
sets={'Original bump 4–8M':[base/f'patch-n{n}.json' for n in [16,24,32]],'Wider bump 4–12M':[P/f'bump-4-12-n{n}.json' for n in [16,24,32]],'Polynomial 4–8M':[P/f'poly6-4-8-n{n}.json' for n in [16,24,32]],'Polynomial 4–12M':[P/f'poly6-4-12-n{n}.json' for n in [16,24,32,48]]}
summary={};fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for label,paths in sets.items():
 rows=[json.loads(x.read_text()) for x in paths];n=np.array([x['n'] for x in rows]);e={f:np.array([x['variants'][0]['regions']['exterior'][f]['rms'] for x in rows]) for f in ['H','M']}
 summary[label]={'N':n.tolist(),'norms':{f:v.tolist() for f,v in e.items()},'orders':{f:(np.log(v[:-1]/v[1:])/np.log(n[1:]/n[:-1])).tolist() for f,v in e.items()},'horizon':rows[0].get('initial_horizon',.0629244910361)}
 for j,f in enumerate(['H','M']):
  axs[0,j].loglog(n,e[f],'o-',label=label);axs[1,j].plot(n[1:],summary[label]['orders'][f],'o-',label=label)
for j,f in enumerate(['H','M']):
 axs[0,j].set(title=f'Full 3D initial {f} RMS',xlabel='N',ylabel='RMS, fixed r > 0.063');axs[1,j].axhline(4,color='k',ls=':',label='Order four');axs[1,j].set(xlabel='Finer N in pair',ylabel='Observed order',ylim=(0,5));axs[0,j].grid(alpha=.25,which='both');axs[1,j].grid(alpha=.25)
axs[0,0].legend(fontsize=8);fig.suptitle('Initial slicing-profile controls — original restoring collar unchanged')
fig.savefig(P/'initial-convergence.png',dpi=180);plt.close(fig)
controls=json.loads((P/'poly6-4-12-n16-controls.json').read_text());r=controls['variants'];summary['patch_controls']={v['name']:{f:v['regions']['exterior'][f]['rms']/r[0]['regions']['exterior'][f]['rms']-1 for f in ['H','M']} for v in r}
summary['N32_improvement_factors']={f:summary['Original bump 4–8M']['norms'][f][2]/summary['Polynomial 4–12M']['norms'][f][2] for f in ['H','M']}
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
