"""Plot read-only BC2 native point samples; do not smooth or infer convergence."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'completed';a=np.genfromtxt(P/'endpoint-slices.csv',delimiter=',',names=True);meta=json.loads((P/'endpoint-slices.json').read_text());assert meta['schema']=='HYP3D_MPI_BC2' and meta['t_over_M']==10 and meta['native_volume_integral_relative_difference']<1e-9
r=json.loads((P/'puncture.report.json').read_text());track=r['puncture_position'][0]
core=np.sort(a[a['ray']==2],order='x');summary={'core_line_yz':[float(core['y'][0]),float(core['z'][0])],'core_rows':len(core),'profiles':{},'qualification':'Only BC2 admitted10M native point samples; no BG1 admitted endpoint exists. Lines/rays are offset from coordinate axes. No interpolation/smoothing or resolution-convergence claim.'}
fig,axes=plt.subplots(2,3,figsize=(13,7))
for ax,k in zip(axes.flat,['chi','alpha','metric_rr','beta_dev_r','D_r','H']):
 ax.plot(core['x'],core[k],'.-',ms=3);ax.axvline(track,ls='--',color='r',label='Tracker');ax.set(xlabel='compact x',title=k);ax.grid(alpha=.2)
summary['core_fields']={k:dict(min=float(core[k].min()),max=float(core[k].max())) for k in ['chi','alpha','metric_rr','H','Mnorm','D_r']}
axes[0,0].legend();fig.suptitle('BC2 at10M: unsmoothed Cartesian line y=z=-0.0028125\nRadial projections use the origin; their direction reverses across x=0');fig.tight_layout();fig.savefig(P/'core-profiles.png',dpi=170);plt.close(fig)
for file,fields,title in [('shell-fields.png',['chi','alpha','metric_rr','beta_dev_r','D_r','Lambda_r'],'Fields'),('shell-derivatives-constraints.png',['dr_chi','dr_alpha','dr_beta_dev_r','dr_D_r','H','Mnorm'],'Derivatives and constraints')]:
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for ray,label,color in [(0,'near +x face center','#2166ac'),(3,'near -x face center','#b2182b'),(1,'near +x face corner','#4d9221')]:
  z=np.sort(a[(a['ray']==ray)&(a['i']<16)],order='r');summary['profiles'][str(ray)]={k:float(np.max(abs(z[k]))) for k in fields}
  for ax,k in zip(axes.flat,fields):ax.plot(z['r'],z[k],'.-',ms=2.5,lw=1,color=color,label=label)
 for ax,k in zip(axes.flat,fields):
  for x in [.305,.45,.6,.75,.875]:ax.axvline(x,ls=':',color='gray',lw=.7)
  ax.set(xlabel='compact radius r',title=k);ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle('BC2 admitted10M endpoint: '+title+'\nNative shell centers; angular differences are physical/numerical directional differences, not a convergence test');fig.tight_layout();fig.savefig(P/file,dpi=170);plt.close(fig)
summary['csv_sha256']=hashlib.sha256((P/'endpoint-slices.csv').read_bytes()).hexdigest();(P/'profiles-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
