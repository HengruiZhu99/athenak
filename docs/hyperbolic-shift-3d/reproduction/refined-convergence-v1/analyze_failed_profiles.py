"""Forensic profiles, not constraints on an admitted endpoint."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
RAW=ROOT/'output/refined-convergence/failed-field-profiles'
OUT=HERE/'stationary20-separated-failure'/'profiles';OUT.mkdir(exist_ok=True)
a=np.loadtxt(RAW/'profiles.csv',delimiter=',',skiprows=1)
names=(RAW/'profiles.csv').open().readline().strip().split(',');col={n:i for i,n in enumerate(names)}
field=lambda x,n:x[:,col[n]]
failure=ROOT/'output/refined-convergence/frame-separation-aurora/campaigns/stationary20-separated-n16-segment01/output/state.failed-stage.bin'
u=np.memmap(failure,dtype='<f8',mode='r').reshape(-1,31)
f=a[field(a,'state')==1];initial=a[field(a,'state')==0]
indices=field(f,'cell').astype(int);v=u[indices]
bar_rr=field(f,'metric_rr')/field(f,'chi')
bar_t=(3+v[:,1]+v[:,4]+v[:,6]-field(f,'metric_rr'))/(2*field(f,'chi'))
summaries=[]
for layer in np.unique(field(f,'layer')):
 for i in range(16):
  q=(field(f,'layer')==layer)&(field(f,'i')==i);z=f[q];row=dict(layer=int(layer),i=i,r=float(field(z,'r').mean()))
  for name in ['chi','alpha','metric_rr','Theta','D_r']:
   x=field(z,name);row[name]=dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std()))
  row['chi_relative_angular_range']=float(np.ptp(field(z,'chi'))/field(z,'chi').mean())
  row['bar_metric_rr_range']=[float(bar_rr[q].min()),float(bar_rr[q].max())]
  row['bar_metric_tangent_range']=[float(bar_t[q].min()),float(bar_t[q].max())];summaries.append(row)
def ray(x,corner=False):
 q=(field(x,'face')==0)&(field(x,'j')==(15 if corner else 8))&(field(x,'k')==(15 if corner else 8))
 return x[q][np.argsort(field(x[q],'r'))]
c=ray(f);e=ray(f,True);r=field(c,'r');ref=ray(initial)
fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for ax,name,label in zip(axs.flat,['chi','alpha','metric_rr','beta_dev_r','D_r','Theta'],['chi','Conformal lapse','Conformal BSSN metric rr','(beta + x) radial','D radial','Theta']):
 ax.plot(field(ref,'r'),field(ref,name),color='gray',label='SERIAL two-step reference')
 ax.plot(r,field(c,name),'o-',ms=2,label='Failed stage: face center')
 ax.plot(field(e,'r'),field(e,name),'--',label='Failed stage: face corner')
 if name in ['chi','metric_rr']:ax.set_yscale('log')
 ax.set(xlim=(.6,1),xlabel='Compact radius r',ylabel=label);ax.grid(alpha=.2)
 for b in [.65,.78,.875,.90,.925,.95]:ax.axvline(b,color='gray',lw=.5,alpha=.3)
axs[0,0].legend(fontsize=7)
fig.suptitle('Failed stationary20 stage: large radial changes, small angular variation\nNominal shell samples; no failed surface available; no admitted endpoint')
fig.savefig(OUT/'radial-fields.png',dpi=170);fig.savefig(OUT/'radial-fields.pdf');plt.close(fig)
last=[x for x in summaries if x['layer']==11 and x['i']==15][0]
record=dict(last_radial_cell=last,maximum_relative_angular_chi_range=max(x['chi_relative_angular_range'] for x in summaries),
 last_center_chi_ratio=float(field(c,'chi')[-1]/field(ref,'chi')[-1]),
 last_center_adjacent_chi_ratio=float(field(c,'chi')[-1]/field(c,'chi')[-2]),
 qualification='Failed-stage geometry is not admitted. Angular ranges are unweighted nominal-point ranges at fixed radius. Near-center/corner rays are distinct angular lines, not exact symmetry axes. Radial changes do not by themselves prove continuum slicing pathology or resolution sufficiency.',
 profiles_sha256=hashlib.sha256((RAW/'profiles.csv').read_bytes()).hexdigest(),
 failed_active_sha256=hashlib.sha256(failure.read_bytes()).hexdigest())
(OUT/'summary.json').write_text(json.dumps(record,indent=2)+'\n');(OUT/'radial-statistics.json').write_text(json.dumps(summaries,indent=2)+'\n')
for n in ['build.json','run.log','profiles.json']:shutil.copy2(RAW/n,OUT/n)
np.savetxt(OUT/'face-center.csv',c,delimiter=',',header=','.join(names),comments='')
np.savetxt(OUT/'face-corner.csv',e,delimiter=',',header=','.join(names),comments='')
np.savetxt(OUT/'two-step-face-center.csv',ref,delimiter=',',header=','.join(names),comments='')
print(json.dumps(record,indent=2))
