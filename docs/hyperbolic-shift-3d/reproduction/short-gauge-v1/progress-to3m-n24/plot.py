"""Plot measured SG1 N16/N24 histories through3M without interpolation."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
snapshot=json.loads((HERE/'snapshot.json').read_text())
assert hashlib.sha256((HERE/'n24.history.csv').read_bytes()).hexdigest()==snapshot['sha256']
p16=HERE.parent/'n16-completed/puncture.history.csv'
w16=json.loads((HERE.parent/'n16-completed/worker-status.json').read_text())
h16=next(v['sha256'] for k,v in w16['cases'][0]['artifacts'].items() if k.endswith('.history.csv'))
assert hashlib.sha256(p16.read_bytes()).hexdigest()==h16
histories={16:np.atleast_1d(np.genfromtxt(p16,delimiter=',',names=True)),24:np.atleast_1d(np.genfromtxt(HERE/'n24.history.csv',delimiter=',',names=True))}
fig,axes=plt.subplots(2,3,figsize=(14,8))
fields=[('H_exterior_L2','Exterior Hamiltonian RMS'),('M_exterior_L2','Exterior momentum magnitude RMS'),('Mr_exterior_L2','Exterior radial momentum RMS'),('H_L2','Full-domain Hamiltonian RMS'),('M_L2','Full-domain momentum magnitude RMS'),('horizon_irreducible_mass','Accepted horizon mass error (%)')]
result=dict(scope=snapshot['scope'],n24_job=snapshot['job'],n24_snapshot_sha256=snapshot['sha256'],n16_history_sha256=h16,cutoffs='Each run uses its own complete candidate horizon; not a common-cutoff comparison.',resolutions={})
for n,color in [(16,'#b55315'),(24,'#1769aa')]:
 data=histories[n];data=data[data['t_over_M']<=3+1e-10];assert abs(data['t_over_M'][-1]-3)<1e-9
 for ax,(field,title) in zip(axes.flat,fields):
  values=data[field]
  if field=='horizon_irreducible_mass':values=np.where(data['horizon_found']==1,100*(3*values-1),np.nan)
  ax.plot(data['t_over_M'],values,'.-',label=f'N{n}',color=color,ms=5)
  ax.set(xlabel='t / M',ylabel=title,xlim=(0,3));ax.grid(alpha=.2)
 for ax in axes[1,:2]:ax.set_yscale('log')
 last=data[-1];accepted=data[(data['horizon_found']==1)&np.isfinite(data['horizon_irreducible_mass'])]
 result['resolutions'][n]=dict(rows=len(data),accepted_horizon_rows=int(np.sum(data['horizon_found']==1)),
  endpoint={key:float(last[key]) if np.isfinite(last[key]) else None for key in ['t_over_M','H_exterior_L2','M_exterior_L2','Mr_exterior_L2','H_L2','M_L2','exterior_cutoff','horizon_found','horizon_irreducible_mass','null_surface','null_ghost']},
  last_accepted_mass=dict(t_over_M=float(accepted[-1]['t_over_M']),relative_error=float(3*accepted[-1]['horizon_irreducible_mass']-1)) if len(accepted) else None)
axes[0,0].legend();fig.suptitle('SG1: measured N16 / N24 histories through 3M\nFresh direct chi/lapse; same gauge and dt/dx; N24 run remains in progress')
fig.tight_layout(rect=(0,.08,1,.93))
fig.text(.5,.025,'Exterior norms use each run’s own candidate horizon. Missing accepted masses remain gaps.\nFull-domain panels use logarithmic scales to retain the puncture transient. No missing-value interpolation.',ha='center',fontsize=10)
fig.savefig(HERE/'histories.png',dpi=150)
result['n24_over_n16_at3m']={key:result['resolutions'][24]['endpoint'][key]/result['resolutions'][16]['endpoint'][key] for key in ['H_exterior_L2','M_exterior_L2','Mr_exterior_L2']}
(HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
