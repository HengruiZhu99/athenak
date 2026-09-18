#!/usr/bin/env python3
"""Plot native SMR cells, without globally upsampling a highly refined mesh.

Residual files are x-z cuts (slice_x2=0); fluid files are x-y cuts
(slice_x3=0). The selected plane is represented by actual cell centers.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'vis/python'))
import bin_convert


def plot(files, variable, bounds, output, signed=False):
    datasets = [bin_convert.read_binary(str(f)) for f in files]
    values = [np.abs(v) for d in datasets for v in d['mb_data'][variable]]
    vmax = max(np.max(v) for v in values)
    # A color-scale floor only; the plotted data and simulations are unchanged.
    vmin = max(vmax*1e-8, 1e-30)
    norm = (SymLogNorm(linthresh=max(vmax*1e-3,1e-30),vmin=-vmax,vmax=vmax)
            if signed else LogNorm(vmin=vmin, vmax=max(vmax, vmin*10)))
    fig, axes = plt.subplots(1, len(files), figsize=(4*len(files), 4),
                             squeeze=False, layout='constrained')
    fluid = variable == 'dens'
    vertical = 1 if fluid else 2
    for ax, d in zip(axes[0], datasets):
        for geom, index, v in zip(d['mb_geometry'], d['mb_index'], d['mb_data'][variable]):
            n = [d['nx1_mb'], d['nx2_mb'], d['nx3_mb']]
            normal = 2 if fluid else 1
            cut_index = 0
            if v.shape[2-normal]>1:
                if not geom[2*normal]<=0<geom[2*normal+1]:continue
                dx=(geom[2*normal+1]-geom[2*normal])/n[normal]
                cut_index=int((0-geom[2*normal])/dx)-int(index[2*normal])
            edges = [geom[2*a] + (index[2*a] + np.arange(v.shape[2-a]+1))
                     *(geom[2*a+1]-geom[2*a])/n[a] for a in range(3)]
            if (edges[0][-1] < bounds[0] or edges[0][0] > bounds[1] or
                    edges[vertical][-1] < bounds[2] or edges[vertical][0] > bounds[3]):
                continue
            cut = v[cut_index,:,:] if fluid else v[:,cut_index,:]
            im = ax.pcolormesh(edges[0], edges[vertical],
                               cut if signed else np.ma.masked_less_equal(abs(cut), 0), norm=norm,
                               cmap='RdBu_r' if signed else 'magma' if fluid else 'viridis', rasterized=True)
        if bounds[0] < 1 and bounds[1] > -1:
            ax.add_patch(plt.Circle((0,0), 1, fill=False, edgecolor='white',
                                   linewidth=1, linestyle='--'))
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='x [M]',
               ylabel=('y' if fluid else 'z')+' [M]',
               title=f't = {d["time"]:.3g} M', aspect='equal')
    fig.colorbar(im, ax=axes[0].tolist(), label='Density [BH units]' if fluid else 'Theta' if signed else '|Theta|')
    fig.suptitle('Native mesh slice: '+('stellar density' if fluid else 'residual constraint field'))
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', type=Path, nargs='+')
    p.add_argument('--variable', choices=['dens','z4c_Theta'], required=True)
    p.add_argument('--bounds', type=float, nargs=4, required=True,
                   metavar=('XMIN','XMAX','VMIN','VMAX'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--signed',action='store_true',help='Diverging symmetric-log scale for signed Theta')
    a = p.parse_args()
    if a.signed and a.variable!='z4c_Theta':p.error('--signed is for Theta only')
    plot(a.files, a.variable, a.bounds, a.output, a.signed)
