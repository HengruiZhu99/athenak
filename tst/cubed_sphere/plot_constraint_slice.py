#!/usr/bin/env python3
"""Plot equatorial native constraint magnitudes from diagnostic-only cell dumps."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle,Rectangle
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained',squeeze=False);panels=[]
for row,name in enumerate(['puncture']):
 d=np.genfromtxt(a.root/name/'constraint_cells.csv',delimiter=',',names=True);meta=json.loads((a.root.parent/'long-stage02/summary.json').read_text());c=meta['cases'][name];L=meta['configurations'][name]['extra'].get('mesh/x1max',.25)
 # Match the two adjacent equatorial cell planes in each logical patch.
 keep=np.zeros(len(d),dtype=bool)
 for region in [0,1]:
  for m in np.unique(d['m'][d['region']==region]):
   ids=np.where((d['region']==region)&(d['m']==m))[0]
   if region==0:
    zmin=np.min(np.abs(d['z'][d['region']==0]));keep[ids[np.abs(d['z'][ids])<zmin+1e-12]]=True
   elif int(m)//2 in [0,1]:
    index=d['k'][ids] if int(m)//2==0 else d['j'][ids];mid=(index.min()+index.max())/2;keep[ids[np.abs(index-mid)<1]]=True
 selected=d[keep];sources=[]
 for region in [0,1]:
  q=selected[selected['region']==region];xy=np.round(np.column_stack([q['x'],q['y']]),12);points,inv=np.unique(xy,axis=0,return_inverse=True);count=np.bincount(inv)
  assert np.all(count==2),'expected paired planes'
  values={k:np.bincount(inv,weights=np.abs(q[k]))/count for k in ['H','M','Z']}
  tr=tri.Triangulation(points[:,0],points[:,1]);cent=points[tr.triangles].mean(axis=1)
  if region==1:tr.set_mask(np.max(abs(cent),axis=1)<L)
  sources.append((tr,values))
 horizon=None;hp=a.root/name/'cubed_sphere_horizon_candidates.csv'
 if hp.exists():
  hs=list(csv.DictReader(hp.open()));horizon=float(hs[-1]['r']) if hs and hs[-1]['found']=='1' else None
 for col,key in enumerate(['H','M','Z']):
  ax=axes[row,col];vmin=1e-10 if key!='Z' else 1e-12;vmax=10 if key!='Z' else .1
  for tr,v in sources:im=ax.tripcolor(tr,np.maximum(v[key],np.finfo(float).tiny),shading='flat',cmap='magma',norm=LogNorm(vmin,vmax))
  ax.add_patch(Rectangle((-L,-L),2*L,2*L,fill=False,edgecolor='cyan',lw=.9))
  ax.add_patch(Circle((0,0),.98,fill=False,edgecolor='.3',lw=1))
  if horizon:ax.add_patch(Circle((0,0),horizon,fill=False,edgecolor='lime',ls='--',lw=1))
  ax.set(xlim=(-1.01,1.01),ylim=(-1.01,1.01),aspect='equal',xlabel='x (compactified coordinate)',ylabel='y (compactified coordinate)')
  ax.set_title(f"{name.capitalize()}, t={float(c['last_rows'][-1]['time']):.5f}: "+{'H':r'$|H|$','M':r'$\|M\|_{\rm conformal}$','Z':r'$\|Z\|_{\rm conformal}$'}[key]);fig.colorbar(im,ax=ax,shrink=.83,extend='both')
  panels.append(dict(case=name,field=key,slab_min=float(min(v[key].min() for _,v in sources)),slab_max=float(max(v[key].max() for _,v in sources))))
fig.suptitle('Native constraints — read-only CPU reconstruction of the GPU puncture checkpoint\nAdjacent ±z cell magnitudes averaged; cyan: bulk–shell interface; green dashed: spherical horizon candidate',fontsize=13)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);fig.savefig(a.output.with_suffix('.pdf'));a.output.with_suffix('.json').write_text(json.dumps(dict(panels=panels,note='Linear equatorial averaging of magnitudes; planar triangular shading. Empty rim has no cell-center data. Same field color limits for both cases; saturated values are visualization only, never evolution clipping.'),indent=2)+'\n')
