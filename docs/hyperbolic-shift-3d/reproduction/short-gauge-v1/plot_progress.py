"""Measured early histories; live SG1 snapshot is not a terminal verification."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;out=HERE/'progress-to3m'
paths={'IT1':HERE.parent/'inner-telegrapher-v1/n16-completed/puncture.history.csv','DO1':HERE.parent/'direct-outer-v1/n16-completed/puncture.history.csv','DG1':HERE.parent/'direct-outer-investigation-v1/dg1-completed/puncture.history.csv','LG1':HERE.parent/'late-gauge-v1/n16-failed/puncture.history.csv','SG1':out/'puncture.history.csv'}
labels={'IT1':'Earlier gauge (IT1)','DO1':'Wide blend (DO1)','DG1':'Wide blend, restored damping (DG1)','LG1':'Outward blend (LG1, failed)','SG1':'Original interval, new driver (SG1)'}
colors={'IT1':'#777777','DO1':'#b55315','DG1':'#1769aa','LG1':'#228844','SG1':'#8b3aa3'}
fields=[('H_exterior_L2','Exterior Hamiltonian RMS'),('M_exterior_L2','Exterior momentum RMS'),('Theta_exterior_L2','Exterior Theta RMS'),('horizon_irreducible_mass','Accepted irreducible mass / M − 1')]
fig,axes=plt.subplots(2,2,figsize=(11,7));records={}
for label,p in paths.items():
 d=np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True));d=d[d['t_over_M']<=3+1e-9]
 assert len(d)>0 and np.all(np.diff(d['t_over_M'])>0)
 if label=='SG1':assert abs(d[-1]['t_over_M']-3)<1e-9
 for ax,(key,title) in zip(axes.flat,fields):
  values=np.where(d['horizon_found']==1,3*d[key]-1,np.nan) if key=='horizon_irreducible_mass' else d[key]
  ax.plot(d['t_over_M'],values,label=labels[label],color=colors[label],marker='.',markersize=3,linewidth=2 if label=='SG1' else 1.3)
  ax.set(xlabel='t / M',ylabel=title,xlim=(0,3.02));ax.grid(alpha=.2)
  if key!='horizon_irreducible_mass':ax.set_yscale('log')
 last=d[-1];records[label]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),last_t_over_M=float(last['t_over_M']),rows=len(d),accepted_horizon_rows=int(np.sum(d['horizon_found'])),last_values={k:float(last[k]) if np.isfinite(last[k]) else None for k in ['H_L2','H_exterior_L2','M_exterior_L2','Theta_exterior_L2','horizon_found','horizon_trial_radius','horizon_irreducible_mass']})
axes[0,0].legend(fontsize=7);fig.suptitle('N16 gauge comparison: measured evolution through 3M')
fig.tight_layout(rect=(0,.09,1,.96));fig.text(.5,.02,'SG1 is a live-run snapshot; reaching 3M is not the 10M accuracy gate. LG1 ends at its last diagnostic, 2.85M.\nExterior norms use each complete candidate horizon. Rejected mass measurements remain gaps.',ha='center',fontsize=9)
fig.savefig(out/'comparison.png',dpi=160)
result=dict(scope='Provisional SG1 history through3M, not terminal artifact verification; no interpolation or extrapolation',cases=records,n24_admitted=False)
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['last_values'] for k,v in records.items()},indent=2))
