"""Nominal block section of the verified combined mesh (not a cell drawing)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Wedge,Patch
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
verify=json.loads((D/'sealed-topology-controls/verification.json').read_text());assert verify['passed']
a=json.loads((D/'sealed-topology-controls/combined-analysis.json').read_text());assert a['old_checkpoint']['leaves']==a['new_checkpoint']['leaves']==400
blocks=json.loads((D/'smr-tube-checks/topology-mpi1.json').read_text())['blocks'];assert len(blocks)==400
cfg={}
for line in (R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial24_angular2_reference.cfg').read_text().splitlines():
 line=line.split('#')[0].strip()
 if '=' in line:
  k,v=line.split('=',1);cfg[k.strip()]=v.strip()
radii=[.35]+list(map(float,cfg['shell_radii'].split(',')));factors=list(map(int,cfg['shell_angular_subdivisions'].split(',')))
assert len(factors)==24 and sum(6*x*x for x in factors)==432
fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained');colors=plt.get_cmap('Blues')
for ax in axes:
 for b in blocks:
  if b['lo'][2]<=0<b['hi'][2]:ax.add_patch(Rectangle((b['lo'][0],b['lo'][1]),b['hi'][0]-b['lo'][0],b['hi'][1]-b['lo'][1],facecolor=colors(.15+.7*b['level']/3),edgecolor='.35',lw=.35))
 ax.set(aspect='equal',xlabel='Compact x',ylabel='Compact y')
ax=axes[0]
ax.add_patch(Wedge((0,0),.95,0,360,width=.17,facecolor='#f7ce9a',alpha=.45,zorder=-1))
for lo,hi,f in zip(radii[:-1],radii[1:],factors):
 ax.add_patch(Circle((0,0),hi,fill=False,edgecolor='#91591f' if f==2 else '.5',lw=.65))
 for theta in np.linspace(np.pi/4,2*np.pi+np.pi/4,4*f,endpoint=False):
  ax.plot(np.array([lo,hi])*np.cos(theta),np.array([lo,hi])*np.sin(theta),color='#91591f' if f==2 else '.5',lw=.65)
ax.add_patch(Circle((0,0),.35,fill=False,color='.5',lw=.65))
ax.set(xlim=(-1.04,1.04),ylim=(-1.04,1.04),title='400 Cartesian leaves + 432 shell blocks\nScri at r=1: original angular grid')
ax.legend(handles=[Patch(facecolor=colors(.5),edgecolor='.35',label='Nominal Cartesian blocks'),Patch(facecolor='#f7ce9a',edgecolor='#91591f',label='Refined transition shells: 0.78–0.95')],loc='lower center',fontsize=8)
ax=axes[1]
row=verify['full_smr_final'];rad=row['horizon_max_radius'];buffer=row['finest_stencil_buffer']
ax.add_patch(Circle((row['tracker'][0],row['tracker'][1]),rad,fill=False,color='#bd342c',lw=1.6,label='Final short-test horizon enclosure'))
ax.add_patch(Circle((row['tracker'][0],row['tracker'][1]),rad+buffer,fill=False,color='#bd342c',ls='--',lw=1.1,label='Enclosure + four finest cells'))
ax.set(xlim=(-.16,.16),ylim=(-.105,.105),title='Central SMR tube; finest spacing 0.0020833')
ax.legend(loc='lower center',fontsize=8)
fig.suptitle('Verified N12 boosted candidate: block boundaries, not individual cells\nNominal Cartesian/shell overlap retained; short test reaches only 0.00025M',fontsize=12)
O=D/'sealed-topology-controls'
for ext in ['png','pdf']:fig.savefig(O/('combined-grid.'+ext),dpi=175)
