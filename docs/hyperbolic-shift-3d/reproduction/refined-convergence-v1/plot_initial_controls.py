"""Plot preserved native-profile and initial spacelike-margin controls."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists());OUT=ROOT/'output/refined-convergence'
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
rows=[]
for name,label in [('flat','RF1 narrow join'),('central6_stationary','6–18M, plateau .5'),('central6_matched','6–18M, matched plateau'),('central20_stationary','20–40M, plateau .5'),('central20_matched','20–40M, matched plateau')]:
 p=OUT/(name+'-profiles.csv');a=np.genfromtxt(p,names=True,delimiter=',');r=a['compact_r']
 axs[0,0].plot(r,a['omega'],label=label);axs[0,1].plot(r,a['ddomega'],label=label)
 rows.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),max_abs_ddomega=float(np.max(abs(a['ddomega'])))))
for name,label in [('boost-4-12','original4–12M'),('boost-6-18','6–18M'),('boost-20-40','20–40M')]:
 p=OUT/(name+'.csv');a=np.genfromtxt(p,names=True,delimiter=',');r=np.unique(a['r']);q=[];j=[]
 for x in r:
  sel=a[a['r']==x];q.append(np.min(sel['Q']/sel['Q0']));j.append(np.min(sel['detJ']))
 axs[1,0].plot(r,q,label=label);axs[1,1].plot(r,j,label=label)
 rows.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for ax,title in zip(axs.flat,['Conformal frame','Transition curvature','Initial spacelike margin: minimum over8 rays','Initial boost-map Jacobian: minimum over8 rays']):
 ax.set_title(title,fontsize=10);ax.set_xlabel('compact radius r');ax.grid(alpha=.25);ax.legend(fontsize=7)
axs[0,0].set_ylabel('Omega');axs[0,1].set_ylabel("Omega''");axs[1,0].set_ylabel('Q / Q0');axs[1,1].set_ylabel('det J')
fig.suptitle('Initial controls only — no evolved convergence claim')
fig.savefig(HERE/'initial-controls.png',dpi=170);fig.savefig(HERE/'initial-controls.pdf')
(HERE/'initial-controls-inputs.json').write_text(json.dumps(rows,indent=2)+'\n')
