"""Plot native puncture profiles and closest-shell refinement without masking values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_native_puncture import cells

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('summary',type=Path);ap.add_argument('--output',required=True,type=Path)
args=ap.parse_args();rows=sorted(json.loads(args.summary.read_text()),key=lambda r:-r['min_spacing'])
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
colors=plt.cm.viridis(np.linspace(.08,.93,len(rows)))
for row,color in zip(rows,colors):
    run=Path(row['run']);files=sorted((run/'bin').glob('*.bin'))
    data,xyz,dx,u=cells(files[-1]);r=np.linalg.norm(xyz,axis=1)
    if not np.isfinite(u).all(): raise ValueError('Nonfinite native output')
    # Group equal actual sampled radii, retaining the largest angular value.
    radii,inverse=np.unique(np.round(r,12),return_inverse=True)
    for ax,values in zip(axes[0],[u[:,18],np.linalg.norm(u[:,43:46],axis=1)]):
        maxima=np.zeros(len(radii));np.maximum.at(maxima,inverse,np.abs(values))
        ax.plot(radii,maxima,'.-',ms=2,lw=1,color=color,label=f"h=M/{1/row['min_spacing']:g}")
for ax,name in zip(axes[0],['rho','L']):
    ax.set(xscale='log',xlim=(min(r['min_sample_radius'] for r in rows)*.8,.6),
           xlabel='Actual native radius r/M',ylabel=rf'$|{name}|$',title='Native radial envelope at 6M')
for ax,name in zip(axes[1],['rho','L']):
    n=[1/r['min_spacing'] for r in rows]
    ax.plot(n,[r['fields'][name]['max_abs'] for r in rows],'o-',label='Maximum in native slice')
    ax.plot(n,[r['fields'][name]['closest_shell_max'] for r in rows],'s-',label='Closest native shell')
    ax.set(xscale='log',xlabel='M / finest spacing',ylabel=rf'$|{name}|$',title='Core refinement at fixed time')
    ax.legend(fontsize=9)
for ax in axes.flat: ax.grid(alpha=.22)
axes[0,0].legend(fontsize=8,ncol=2)
fig.suptitle('Regular advective PC-GH, lambda=1: native cell samples\nFD2 / RK4, fixed dissipation; slice maxima are not full-volume bounds',fontsize=13)
fig.savefig(args.output,dpi=180)
