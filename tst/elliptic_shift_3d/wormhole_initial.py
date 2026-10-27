#!/usr/bin/env python3
"""Sample the specified isotropic M=1/3 slice on actual composite cell centers.

Only initial data are radial. Coupled evolution never uses this radial oracle.
The coordinate CSV must come from the pilot with the same mesh descriptor.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole
p=argparse.ArgumentParser();p.add_argument('coordinates',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
points=np.genfromtxt(a.coordinates,delimiter=',',names=True)
r=points['r'];unique,inverse=np.unique(r,return_inverse=True)
initial=IsotropicWormhole(mass=1/3,quadrature_order=96)
q,_,_,meta=initial.data(unique)
chi=np.array([m['chi_total'] for m in meta]);alpha=np.array([m['physical_lapse'] for m in meta])*(1-unique**2)/2
values=np.column_stack([unique,chi,alpha,q[:,2],q[:,3]-3])[inverse]
with a.output.open('w') as out:
 out.write(f'ISOTROPIC_M1_3 {len(r)}\n');np.savetxt(out,values,fmt='%.17g')
manifest=dict(mass=initial.mass,k=initial.k,isotropic_transition=initial.r0,throat_radius=initial.k*initial.mass/2,
 minimum_cell_radius=float(min(r)),cells=len(r),unique_radii=len(unique),quadrature_order=96,
 initial_data='existing IsotropicWormhole, Minkowski reference',
 coordinates_sha256=hashlib.sha256(a.coordinates.read_bytes()).hexdigest(),
 initializer_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
 source_sha256=hashlib.sha256((Path(__file__).parents[1]/'elliptic_shift_1d/wormhole_isotropic.py').read_bytes()).hexdigest())
a.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
