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
ap.add_argument('--label',default='Regular advective PC-GH, lambda=1')
ap.add_argument('--numerics',default='FD2 / RK4')
args=ap.parse_args();rows=sorted(json.loads(args.summary.read_text()),key=lambda r:-r['min_spacing'])
common_time=rows[0]['time']
if any(abs(r['time']-common_time)>1e-4 for r in rows):
    raise ValueError('Core refinement comparison requires a common time')
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
colors=plt.cm.viridis(np.linspace(.08,.93,len(rows)))
for row,color in zip(rows,colors):
    run=Path(row['run']);files=sorted((run/'bin').glob('*.bin'))
    source=Path(row['source_file']) if 'source_file' in row else files[-1]
    data,xyz,dx,u=cells(source);r=np.linalg.norm(xyz,axis=1)
    if abs(float(data['time'])-row['time'])>1e-10:
        raise ValueError('Native plot source time differs from its summary')
    if not np.isfinite(u).all(): raise ValueError('Nonfinite native output')
    # Group equal actual sampled radii, retaining the largest angular value.
    radii,inverse=np.unique(np.round(r,12),return_inverse=True)
    for ax,values in zip(axes[0],[u[:,18],np.linalg.norm(u[:,43:46],axis=1)]):
        maxima=np.zeros(len(radii));np.maximum.at(maxima,inverse,np.abs(values))
        ax.plot(radii,maxima,'.-',ms=2,lw=1,color=color,label=f"h=M/{1/row['min_spacing']:g}")
for ax,name in zip(axes[0],['rho','L']):
    ax.set(xscale='log',xlim=(min(r['min_sample_radius'] for r in rows)*.8,.6),
           xlabel='Actual native radius r/M',ylabel=rf'$|{name}|$',title=f'Native radial envelope at {common_time:g}M')
for ax,name in zip(axes[1],['rho','L']):
    n=[1/r['min_spacing'] for r in rows]
    ax.plot(n,[r['fields'][name]['max_abs'] for r in rows],'o-',label='Maximum in native slice')
    ax.plot(n,[r['fields'][name]['closest_shell_max'] for r in rows],'s-',label='Closest native shell')
    ax.set(xscale='log',xlabel='M / finest spacing',ylabel=rf'$|{name}|$',title='Core refinement at fixed time')
    ax.legend(fontsize=9)
for ax in axes.flat: ax.grid(alpha=.22)
axes[0,0].legend(fontsize=8,ncol=2)
fig.suptitle(args.label+': native cell samples\n'+args.numerics+', fixed dissipation; slice maxima are not full-volume bounds',fontsize=13)
fig.savefig(args.output,dpi=180)
