#!/usr/bin/env python3
"""Isotropic hyperboloidal wormhole geometry, mass, lapse and inner end."""
import argparse,json
from types import SimpleNamespace
import numpy as np
from wormhole_isotropic import IsotropicWormhole
from kernel import Kernel
from numerics import RadialOperators
from elliptic import solve
import g1_rhs
p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args()
r=(np.arange(512)+.5)/511.5;w=IsotropicWormhole();q,d,dd,meta=w.data(r)
other=IsotropicWormhole(quadrature_order=128);ref,_,_,_=other.data(r)
coordinate_error=float(np.max(abs(q-ref)));assert coordinate_error<2e-11,coordinate_error
alpha=(1+r*r)/2+q[:,6];chi=np.array([x['chi_total'] for x in meta]);op=RadialOperators(r)
beta,db,ddb,_=solve(op,np.ones_like(r),d[:,1],dd[:,1],alpha,r+d[:,6],q[:,2],d[:,2],-1.)
R,S,diag=Kernel(a.library).parts(r,q,d,dd,-r,-np.ones_like(r),np.zeros_like(r),total_chi=chi)
constraints=np.max(abs(diag[:,:4]),axis=0);assert max(constraints)<2e-10,constraints
bulk=(r>.01)&(r<.98);mass_error=float(max(abs(diag[bulk,10]-.1)));assert mass_error<2e-9,mass_error
outer=np.array([x['branch']=='hyperboloidal' for x in meta])&(r<1)
rates=R[:-1,:6]+S[:-1,:6]/((1-r[:-1]*r[:-1])/2)[:,None]
stationary_native=float(max(abs(rates[outer[:-1]]).ravel()))
assert max(abs(rates[outer[:-1]&(r[:-1]<.98)]).ravel())<2e-9
# The exact elliptic exterior is beta=-r because A_rr vanishes there. Use
# that analytic value here; quadrature stencils straddling the transition have
# a separate finite-resolution error, not a nonstationary exact slice.
exact_outer=SimpleNamespace(r=r,integral=lambda values:np.zeros_like(values))
gR,gS,*_=g1_rhs.parts(exact_outer,q,d,dd,restoring=1)
stable_rates=gR[:-1,:6]+gS[:-1,:6]/((1-r[:-1]*r[:-1])/2)[:,None]
stationary_geometry=float(max(abs(stable_rates[outer[:-1]]).ravel()));assert stationary_geometry<2e-11,stationary_geometry
assert beta[0]/r[0]>0 # All continuum characteristics become inner outflow asymptotically.
assert np.max(abs(q[:,1]))==np.max(abs(q[:,5]))==0
rr=np.array([1e-4,5e-5,2.5e-5]);inner=[w.point(float(x))[3] for x in rr]
radius_product=np.array([z['R'] for z in inner])*rr
limit=w.mass*w.mass*w.k/4
assert max(abs(radius_product/limit-1))<.01
for radius,z in zip(rr,inner):
    rho=radius/w.k;physical_lapse=(1+w.mass/(2*rho))**-2
    np.testing.assert_allclose(z['physical_lapse'],physical_lapse,rtol=2e-15)
record=dict(coordinate_quadrature_change=coordinate_error,constraints=constraints.tolist(),mass_error_bulk=mass_error,
    stationary_exterior_geometric_RHS=stationary_geometry,native_uncanceled_exterior_RHS=stationary_native,
    inner_shift_slope=float(beta[0]/r[0]),r0=w.r0,k=w.k,
    radius_product=radius_product.tolist(),inner_limit=limit,scope='Initial wormhole only; no trumpet evolution or inner parity assumption')
print(json.dumps(record))
