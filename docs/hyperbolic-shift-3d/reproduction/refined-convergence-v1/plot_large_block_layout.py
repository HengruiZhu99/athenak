"""Plot native block bounds and radial cell spacing for the two proposed grids."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
old=json.loads((D/'smr-tube-checks/topology-mpi2.json').read_text())
new=json.loads((D/'large-block-control/block24.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,blocks,n,title in [(axes[0],old['blocks'],12,'12³ blocks; three refinements'),
                           (axes[1],new['core_blocks'],24,'24³ blocks; two refinements')]:
    for b in blocks:
        lo,hi=np.array(b['lo']),np.array(b['hi'])
        if lo[2]<=1e-12<hi[2]:
            ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],fill=False,
                                  edgecolor=plt.cm.viridis(b['level']/3),lw=.65))
    ax.set(xlim=(-.4,.4),ylim=(-.4,.4),xlabel='compact x',ylabel='compact y',title=title)
    ax.set_aspect('equal')
cfg=R/'inputs/hyperbolic_shift_3d/refined_convergence'
for name,n,label in [('central20_boosted_smr_tube_radial24_angular2_reference.cfg',12,'12³ / 24 layers'),
                     ('central20_boosted_block24_smr2_merged13_reference.cfg',24,'24³ / 13 layers')]:
    line=next(s for s in (cfg/name).read_text().splitlines() if s.startswith('shell_radii='))
    radii=np.r_[.35,[float(x) for x in line.split('=')[1].split(',')]]
    axes[2].stairs(np.diff(radii)/n,radii,label=label,lw=1.6)
axes[2].set(yscale='log',xlabel='compact radius r',ylabel='radial cell spacing Δr',title='Radial spacing comparison')
axes[2].axvspan(.78,.95,color='grey',alpha=.12)
axes[2].legend(fontsize=8)
fig.suptitle('Same finest core spacing Δx = 1/480; larger blocks refine a larger volume',fontsize=12)
for ext in ['png','pdf']:fig.savefig(D/f'large-block-control/layout.{ext}',dpi=180)
