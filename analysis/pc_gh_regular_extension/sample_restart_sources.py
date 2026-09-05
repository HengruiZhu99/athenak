"""Sample the reduction source on native double-precision PC-GH checkpoints.

This reader is restricted to single-file, single-rank, vacuum PC-GH checkpoints
with the verified current little-endian ABI. Every size is checked. The source
estimates are frozen FD background samples, not a global nonlinear decay theorem.
"""
import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np
from frozen_reduction_source import configuration, estimates
from make_inputs import parse


def read_checkpoint(path):
    with path.open('rb') as stream:
        prefix = stream.read(65536)
        end = prefix.find(b'<par_end>\n')
        if end < 0:
            raise ValueError('Unrecognized checkpoint parameter header')
        offset = end+10
        params = parse(prefix[:offset].decode())
        if 'pc_gh' not in params or any(k in params for k in ['hydro','mhd','radiation','z4c']):
            raise ValueError('Only vacuum PC-GH checkpoint payloads are supported')
        stream.seek(offset)
        blocks, root_level = struct.unpack('<ii', stream.read(8))
        geometry = np.array(struct.unpack('<9d', stream.read(72)))
        mesh = struct.unpack('<19i', stream.read(76))
        block = struct.unpack('<19i', stream.read(76))
        time, dt, cycle = struct.unpack('<ddi', stream.read(20))
        locations = np.frombuffer(stream.read(blocks*16), dtype='<i4').reshape(blocks,4)
        stream.read(blocks*4)  # Load-balance costs.
        ng = block[0]; counts = np.array(block[1:4])
        shape = (blocks,55,counts[2]+2*ng,counts[1]+2*ng,counts[0]+2*ng)
        per_block = int(np.prod(shape[1:]))*8
        payload_start = path.stat().st_size-blocks*per_block
        if payload_start < stream.tell()+8:
            raise ValueError('Checkpoint dimensions exceed the available payload')
        stream.seek(payload_start-8)
        if struct.unpack('<Q',stream.read(8))[0] != per_block:
            raise ValueError('Checkpoint data-size marker does not match the PC-GH ABI')
    state = np.memmap(path, dtype='<f8', mode='r', offset=payload_start, shape=shape)
    # Include ghost values in this validity check: derivatives below read them.
    for values in state:
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite checkpoint state or ghost value')
    meta = dict(time=time,dt=dt,cycle=cycle,blocks=blocks,ng=ng,counts=counts.tolist(),
                root_level=root_level,geometry=geometry.tolist(),mesh_counts=list(mesh[1:4]))
    return params, meta, locations, state


def sample(path, output, per_shell=24, points=(), extrema=False):
    params, meta, locations, state = read_checkpoint(path)
    counts=np.array(meta['counts']);ng=meta['ng'];geometry=np.array(meta['geometry'])
    order=int(params['pc_gh'].get('spatial_order', 2*(ng-1)))
    if order <= 0:
        order=2*(ng-1)
    if order not in [2,4,6] or ng < order//2+1:
        raise ValueError('Unsupported finite-difference order or insufficient ghosts')
    weights={2:[.5],4:[2/3,-1/12],6:[3/4,-3/20,1/60]}[order]
    root_blocks=np.array(meta['mesh_counts'])/counts
    coordinates, identifiers, spacings = [], [], []
    for m, location in enumerate(locations):
        width=(geometry[3:6]-geometry[:3])/(root_blocks*2.**(location[3]-meta['root_level']))
        lower=geometry[:3]+location[:3]*width
        h=width/counts
        k,j,i=np.indices(tuple(counts[::-1]))
        ijk=np.column_stack([i.ravel(),j.ravel(),k.ravel()])
        coordinates.append(lower+(ijk+.5)*h)
        identifiers.append(np.column_stack([np.full(len(ijk),m),ijk[:,2]+ng,ijk[:,1]+ng,ijk[:,0]+ng]))
        spacings.append(np.tile(h,(len(ijk),1)))
    xyz=np.vstack(coordinates);ids=np.vstack(identifiers);spacing=np.vstack(spacings)
    radius=np.linalg.norm(xyz,axis=1)
    edges=np.geomspace(.999*radius.min(),1.001*radius.max(),25)
    selected=[]
    for left,right in zip(edges[:-1],edges[1:]):
        candidates=np.flatnonzero((radius>=left)&(radius<right))
        if len(candidates):
            selected.extend(candidates[np.linspace(0,len(candidates)-1,min(per_shell,len(candidates)),dtype=int)])
    # Include explicitly requested failure neighborhoods and component extrema;
    # stratified radial samples alone can miss a thin interface instability.
    requested_points=[]
    for point in points:
        nearest=int(np.argmin(np.linalg.norm(xyz-np.asarray(point),axis=1)))
        selected.append(nearest)
        requested_points.append(dict(requested=list(point),sampled=xyz[nearest].tolist()))
    if extrema:
        maxima=np.full(55,-np.inf); winners=np.zeros(55,dtype=int)
        cells_per_block=int(np.prod(counts))
        for m in range(len(locations)):
            active=np.abs(state[m,:,ng:ng+counts[2],ng:ng+counts[1],ng:ng+counts[0]]).reshape(55,-1)
            local=np.argmax(active,axis=1); values=active[np.arange(55),local]
            changed=values>maxima
            maxima[changed]=values[changed]
            winners[changed]=m*cells_per_block+local[changed]
        selected.extend(winners)
    selected=sorted(set(selected))
    rows=[]
    pc=params['pc_gh'];rate=float(pc.get('reduction_rate',0))
    for index in selected:
        m,k,j,i=ids[index];u=np.array(state[m,:,k,j,i]);derivative=np.zeros((3,11));drho=np.zeros(3)
        for d in range(3):
            for offset,weight in enumerate(weights,start=1):
                plus=[k,j,i];minus=[k,j,i];plus[2-d]+=offset;minus[2-d]-=offset
                up=np.array(state[(m,slice(None),*plus)])
                um=np.array(state[(m,slice(None),*minus)])
                derivative[d]+=weight*(configuration(up)-configuration(um))/spacing[index,d]
                drho[d]+=weight*(up[18]-um[18])/spacing[index,d]
        result=estimates(u,derivative[:,8:11],rate,float(pc.get('shift_eta',2)),
                         float(pc.get('shift_switch_z0',.1)),float(pc.get('shift_switch_z1',.5)))
        factorized=u[18]*derivative[:,0]+u[0]*drho
        row=dict(block=int(m),level=int(locations[m,3]-meta['root_level']),
                 x=float(xyz[index,0]),y=float(xyz[index,1]),z=float(xyz[index,2]),
                 r=float(radius[index]),h=float(spacing[index].min()),w=float(u[0]),rho=float(u[18]),
                 alpha=float(u[0]*u[18]),K=float(u[7]),
                 product_derivative_defect=float(np.linalg.norm(derivative[:,7]-factorized)),
                 half_ell_factorized=float(np.linalg.norm(u[43:46]/2-factorized)),
                 half_ell_direct=float(np.linalg.norm(u[43:46]/2-derivative[:,7])),**result)
        rows.append(row)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'source-samples.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(checkpoint=str(path),metadata=meta,samples=len(rows),spatial_order=order,
                requested_points=requested_points,include_component_extrema=extrema,
                scope='Stratified native samples, FD frozen coefficients; not global or uniform puncture bounds',
                minimum_energy_margin=min(rows,key=lambda r:r['energy_rate_margin']),
                maximum_spectral_abscissa=max(rows,key=lambda r:r['frozen_spectral_abscissa']),
                maximum_product_defect=max(rows,key=lambda r:r['product_derivative_defect']))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--per-shell',type=int,default=24)
    ap.add_argument('--point',type=float,nargs=3,action='append',default=[],
                    help='Also sample the nearest active cell to these coordinates')
    ap.add_argument('--extrema',action='store_true',
                    help='Also sample each of the 55 component absolute maxima')
    args=ap.parse_args()
    print(json.dumps(sample(args.checkpoint,args.output,args.per_shell,args.point,args.extrema),indent=2))
