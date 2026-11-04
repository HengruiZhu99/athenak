"""Unsmoothed native-point profiles; retain overlap copies separately for audit."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;DATA=HERE/'convergence-localization';OUT=HERE/'n16-n20-n24'
data={n:np.genfromtxt(DATA/f'n{n}.csv',delimiter=',',names=True) for n in [16,20,24]}
fields=[('H','Hamiltonian'),('Mnorm','Momentum magnitude'),('chi','chi'),('alpha','Conformal lapse'),('dr_beta_dev_r','d_r (beta^r + r)'),('dr_D_r','d_r D^r')]
fig,axs=plt.subplots(6,3,figsize=(13,15),sharex=True)
summary={'input_sha256':{str(n):hashlib.sha256((DATA/f'n{n}.csv').read_bytes()).hexdigest() for n in data},'qualification':'Native sample centers, no interpolation/smoothing. Angular sample directions differ slightly with resolution. Nominal shell cells plotted; evolved overlap copies audited separately. Slice comparisons are not field self-convergence. An overlap mismatch is not by itself a causal instability diagnosis.','overlap':{},'peaks':{}}
for col,ray in enumerate([0,1,3]):
 for n,color in [(16,'#2166ac'),(20,'#d95f02'),(24,'#1b9e77')]:
  a=data[n];b=a[(a['ray']==ray)&(a['i']>=0)&(a['i']<n)&(a['r']>=.2)&(a['r']<=.56)];b=b[np.argsort(b['r'])]
  for ax,(key,label) in zip(axs[:,col],fields):ax.plot(b['r'],b[key],'.-',ms=3,color=color,label=f'N{n}');ax.set_ylabel(label);ax.axvline(.305,color='gray',ls=':');ax.axvline(.45,color='gray',ls=':');ax.grid(alpha=.2)
  q=b[(b['r']>.3)&(b['r']<.4)];summary['peaks'][f'N{n}_ray{ray}']={key:{'max_abs':float(abs(q[key]).max()),'r':float(q[np.argmax(abs(q[key]))]['r'])} for key in ['H','Mnorm','dr_chi','dr_beta_dev_r','dr_D_r']}
 axs[0,col].set_title({0:'+x face, near center',1:'+x face, near corner',3:'−x face, near center'}[ray]);axs[-1,col].set_xlabel('Compact radius r')
axs[0,0].legend();fig.suptitle('BN1 admitted 10M endpoints: inner shell / gauge-transition profiles\nDots are nominal native cells; dotted lines are shell interfaces; no interpolation',y=.995);fig.tight_layout(rect=(0,0,1,.965));fig.savefig(OUT/'slices.png',dpi=150);fig.savefig(OUT/'slices.pdf');plt.close(fig)
for n,a in data.items():
 matches=[]
 for ray in [0,1,3]:
  normal=a[(a['ray']==ray)&(a['i']>=0)&(a['i']<n)]
  extra=a[(a['ray']==ray)&(a['i']>=n)&(a['layer']==0)]
  for e in extra:
   dist=np.sqrt((normal['x']-e['x'])**2+(normal['y']-e['y'])**2+(normal['z']-e['z'])**2);ix=np.argmin(dist)
   if dist[ix]>1e-12:continue
   b=normal[ix];matches.append({'ray':ray,'r':float(e['r']),'coordinate_mismatch':float(dist[ix]),'nominal_layer':int(b['layer']),'extra_layer':int(e['layer']),**{k:{'nominal':float(b[k]),'extra':float(e[k]),'absolute_difference':float(abs(b[k]-e[k]))} for k in ['chi','alpha','metric_rr','beta_r','H','Mnorm','dr_chi','dr_beta_dev_r']}})
 summary['overlap'][str(n)]=matches
(OUT/'slice-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary['peaks'],indent=2))
for n,m in summary['overlap'].items():
 print('OVERLAP',n)
 for ray in [0,1,3]:
  mm=[x for x in m if x['ray']==ray];print(ray,{k:max(x[k]['absolute_difference'] for x in mm) for k in ['chi','alpha','metric_rr','beta_r','H','Mnorm','dr_chi']})
