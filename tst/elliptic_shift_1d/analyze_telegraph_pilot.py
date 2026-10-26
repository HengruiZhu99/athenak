#!/usr/bin/env python3
"""Compare a declared width/KO factorial pilot; no convergence admission."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_telegraph_time import load,numerical_binding,controlled_inputs
from telegraph_analysis_io import trajectory,physical_lapse_gradient

p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.matrix/'input.json').read_text());records=[];base=None
for case in plan['cases']:
 folder=a.matrix/case['name'];m=json.loads((folder/'manifest.json').read_text())
 if m['status'] not in ['finished','failed']:raise ValueError('Wait for terminal pilot runs')
 if m['status']=='finished':
  run=load(folder)
  if base is None:base=run
  else:controlled_inputs(base,run,{'width','ko_order'})
  data=run['data'];ancestry=run['ancestry']
 else:
  numerical_binding(folder,m);data,ancestry=trajectory(folder)
 rows=[]
 for t,d in sorted(data.items()):
  r=d['r'];mask=r<.3;gradient=physical_lapse_gradient(d)[mask]
  h=d['diagnostic'][:,0];i=int(np.argmax(abs(h)))
  rows.append(dict(t=t,H_max=float(abs(h[i])),H_location=float(r[i]),gradient_max=float(max(abs(gradient))),
     gradient_location=float(r[mask][np.argmax(abs(gradient))]),min_inward_logJ=float(min(d['rays'][::2,1]))))
 records.append(dict(name=case['name'],path=str(folder.resolve()),manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
   ancestry=ancestry,status=m['status'],failure=m['failure'],time=m['time'],width=m['arguments']['width'],ko_order=m['arguments']['ko_order'],
   stages=m['all_RK_stages'],history=rows))
a.output.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for rec in records:
 label=f"width {rec['width']}, KO{rec['ko_order']}"+(' (failed)' if rec['status']=='failed' else '')
 for ax,key in zip(axes.flat,['H_max','gradient_max','min_inward_logJ','H_location']):
  ax.plot([r['t'] for r in rec['history']],[r[key] for r in rec['history']],label=label)
for ax,title,ylabel in zip(axes.flat,['Saved-state Hamiltonian maximum','Physical lapse gradient, r < .3','Ingoing characteristic compression','Location of Hamiltonian maximum'],[r'$\max|\mathcal{H}|$',r'$\max|\partial_r\widetilde\alpha|$',r'$\min\log J$','r']):
 ax.set(title=title,xlabel='t',ylabel=ylabel);ax.grid(alpha=.2)
axes[0,0].set_yscale('log');axes[0,0].legend(fontsize=8)
fig.savefig(a.output/'pilot.png',dpi=160);fig.savefig(a.output/'pilot.pdf');plt.close(fig)
result=dict(records=records,scope='One coarse grid, identical continuum parameters and independent quadrature, declared width/KO factorial comparison. Source/binary equality and complete-time coverage verified for completed cases. Failed cases retained as partial trajectories. This pilot cannot establish spatial order, temporal accuracy or goal completion.')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(name=r['name'],status=r['status'],time=r['time'],H_rms=r['stages']['max_accepted_H_rms'],H_max=r['stages']['max_accepted_H_bulk'],scri_pole=r['stages']['max_scri_pole']) for r in records],indent=2))
