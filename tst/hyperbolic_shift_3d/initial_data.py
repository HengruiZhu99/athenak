"""Lift the donor's isotropic hyperboloidal wormhole onto actual 3D cells.

Only initial data use spherical symmetry. Evolution retains all tensor/vector
components. The initial shift is the donor's integral, resolved on 4097 points;
its refinement error is recorded independently from the 3D truncation error.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_1d'))
from puncture import PunctureEvolution
from wormhole_isotropic import IsotropicWormhole

KW=dict(ratio=2.,damping=1.,ko=.5,restoring=4.,inner_shift=0.,
        lapse_floor=1e-10,shift_source='reference',connection_driver='evolved',
        shift_gauge='moving-puncture',gamma_coefficient=4.,grid_stretch=0.,
        shift_inner_stencil='odd',inner_z4_damping=8.)
def initialize(coordinates,library):
    xyz=np.c_[coordinates['x'],coordinates['y'],coordinates['z']]
    radius=coordinates['r']; direction=xyz/radius[:,None]
    unique,inverse=np.unique(radius,return_inverse=True)
    q,_,_,meta=IsotropicWormhole(1/3,96).data(unique)
    q=q[inverse];chi=np.array([m['chi_total'] for m in meta])[inverse]
    lapse=np.array([m['physical_lapse'] for m in meta])[inverse]*(1-radius**2)/2
    shifts=[]
    for n in (2049,4097):
        e=PunctureEvolution(n,library,**KW);initial=e.initial()
        # Smooth beta/r interpolation avoids a spurious nonzero origin value.
        coefficient=np.r_[initial[1,7]/e.r[1],initial[1:,7]/e.r[1:]]
        shifts.append(radius*np.interp(radius,e.r,coefficient))
    out=np.zeros((len(radius),31));out[:,0]=chi;out[:,13:15]=q[:,3:5]
    out[:,18]=lapse-(1+radius**2)/2
    for f,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
        out[:,7+f]=1.5*q[:,2]*(direction[:,i]*direction[:,j]-(i==j)/3)
    out[:,22:25]=shifts[-1][:,None]*direction
    return out,dict(mass=1/3,initial_shift_quadrature_nodes=4097,
                    shift_interpolation_refinement_max=float(np.max(abs(shifts[0]-shifts[1]))),
                    chi_min=float(min(chi)),lapse_min=float(min(lapse)),
                    donor_configuration=KW)
def main():
    p=argparse.ArgumentParser();p.add_argument('--coordinates',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    coords=np.genfromtxt(a.coordinates,delimiter=',',names=True)
    values,metadata=initialize(coords,a.library.resolve())
    with a.output.open('w') as f:
        f.write(str(values.size)+'\n');np.savetxt(f,values,fmt='%.17g')
    metadata.update(coordinates_sha256=hashlib.sha256(a.coordinates.read_bytes()).hexdigest(),
                    library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
                    initial_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
