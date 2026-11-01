"""Compare verified IT1 history with GD1 at the same N16/dt; preserve gaps."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('history',type=Path);a=p.parse_args()
old=HERE.parent/'smooth-direct-v1/n16-completed/puncture.history.csv'
paths=[old,a.history]
fig,axes=plt.subplots(2,2,figsize=(11,7))
fields=[('H_exterior_L2','Exterior Hamiltonian RMS',1),('Mr_exterior_L2','Exterior radial-momentum RMS',1),('horizon_areal_radius','Accepted horizon areal radius / M',3),('null_surface','Running maximum scri null residual',1)]
for path,label,color in zip(paths,['Previous smooth gauge, direct from t=0','Inner telegrapher, direct from t=0'],['#777777','#1769aa']):
 d=np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True));d=d[d['t_over_M']<=10+1e-9]
 for ax,(key,title,scale) in zip(axes.flat,fields):
  ax.plot(d['t_over_M'],scale*d[key],'-o',ms=2,label=label,color=color)
  ax.set(xlabel='t / M',ylabel=title,xlim=(0,10));ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('N16: inner-gauge/damping change; same grid, timestep, floors and outer closure')
fig.tight_layout(rect=(0,.08,1,.95))
fig.text(.5,.025,'Measured samples only; no gap interpolation. Exterior norms use each run’s complete horizon candidate.\nHorizon acceptance flags remain in the source histories. This comparison does not establish convergence.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(HERE/f'comparison.{ext}',dpi=170)
(HERE/'comparison-sources.json').write_text(json.dumps({str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},indent=2)+'\n')
