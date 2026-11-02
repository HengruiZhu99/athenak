"""Render the preserved matching10M diagnostics; no evolution or resampling."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
files={'SG1':p.parents[2]/'short-gauge-n32-v1/comparison/n16-radial.json','A':p.parent/'a-versus-sg1/a-radial.json','C':p.parent/'abc-versus-sg1/c-radial.json','D':p/'d-radial.json'}
fig,axes=plt.subplots(1,2,figsize=(11,4));summary={}
for label,file in files.items():
    d=json.loads(file.read_text());assert d['cutoff']==.1099 and d['t_over_M']==10
    rows=d['radial_cumulative'];r=np.array([s['cutoff'] for s in rows]);volume=-np.diff([s['volume'] for s in rows])
    exterior=next(s for s in rows if s['cutoff']==.1099);regions={}
    for lo,hi in [(.16,.3),(.775,.9)]:
        below=next(s for s in rows if s['cutoff']==lo);above=next(s for s in rows if s['cutoff']==hi)
        regions[f'{lo}-{hi}']={k:(below[k]-above[k])/exterior[k] for k in ['H2','M2','Mr2']}
    summary[label]={'cutoff':d['cutoff'],'norms':d['norms'],'regional_fraction_of_exterior_squared_residual':regions}
    for ax,k in zip(axes,['H2','M2']):
        value=np.sqrt(np.maximum(-np.diff([s[k] for s in rows]),0)/np.maximum(volume,1e-30))
        keep=r[:-1]>=.1099
        ax.stairs(value[keep],r[np.r_[keep,True]],label=label,linestyle='--' if label=='A' else '-')
        ax.set(xlim=(.1099,1),xlabel='compactified radius r',ylabel=k[:-1]+' regional RMS',yscale='log');ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('Scri repair D versus controls at10M; common cutoff r >0.1099');fig.tight_layout()
fig.savefig(p/'radial.png',dpi=160);plt.close(fig)
(p/'radial-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
