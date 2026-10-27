#!/usr/bin/env python3
"""Exact embedded-slice scalar/curvature jets; independently refined shift integral."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole
from numerics import RadialOperators
from radial_reference import interpolate
p=argparse.ArgumentParser();p.add_argument('coordinates',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
coordinates=np.genfromtxt(a.coordinates,delimiter=',',names=True);r=np.minimum(coordinates['r'],1.)
unique,inverse=np.unique(r,return_inverse=True);initial=IsotropicWormhole(mass=1/3)
q,d,dd,meta=initial.data(unique);chi=np.array([m['chi_total'] for m in meta]);alpha=np.ones_like(unique)
alpha[:-1]=np.array([m['physical_lapse'] for m in meta[:-1]])*(1-unique[:-1]**2)/2
betas=[]
for n in [4097,8193]:
 rr=np.linspace(0,1,n);qq,_,_,mm=initial.data(rr[1:]);aa=np.r_[0.,np.array([m['physical_lapse'] for m in mm[:-1]])*(1-rr[1:-1]**2)/2,1.]
 integrand=np.r_[0.,1.5*aa[1:]*qq[:,2]/rr[1:]]
 op=RadialOperators(rr,width=7,puncture=True,allow_puncture_node=True)
 ratio=-1-op.tail_integral(integrand);betas.append(interpolate(rr,ratio,unique))
ratio=betas[-1];beta=unique*ratio;bp=ratio+1.5*alpha*q[:,2]
bpp=1.5*((unique+d[:,6])*q[:,2]+alpha*d[:,2]+alpha*q[:,2]/unique)
data=np.column_stack([unique,chi,d[:,0],dd[:,0],alpha,unique+d[:,6],1+dd[:,6],q[:,2],d[:,2],q[:,3]-3,d[:,3],beta,bp,bpp])[inverse]
with a.output.open('w') as out:
 out.write(f'EXACT_WORMHOLE_V1 {len(r)}\n')
 np.savetxt(out,np.column_stack([coordinates['kind'],coordinates['id'],data]),fmt=['%d','%d']+['%.17g']*14)
print(json.dumps(dict(points=len(r),unique_radii=len(unique),shift_ratio_4097_8193_max=float(max(abs(betas[0]-betas[1]))))))
