"""Finite-radius Cartesian ADM mass and axial Killing angular momentum on native vertices."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from diagnostics import read_binary, reflected_equator


def interpolation(q, count):
    first = np.clip(int(np.floor(q))-1,0,count-4)
    nodes = first+np.arange(4)
    w, d = np.ones(4), np.zeros(4)
    for j in range(4):
        others = nodes[nodes!=nodes[j]]
        denom = np.prod(nodes[j]-others)
        w[j] = np.prod(q-others)/denom
        d[j] = sum(np.prod(q-np.delete(others,k)) for k in range(3))/denom
    return slice(first,first+4), w, d


def charges(frame, radius, ntheta=128):
    bounds = np.asarray(frame["mb_geometry"])
    half = bounds[:,2].min() == 0
    if half and not reflected_equator(frame):
        raise ValueError("northern coverage requires equatorial reflection")
    if radius >= min(bounds[:,1].max(), bounds[:,3].max(), np.inf if half else -bounds[:,2].min()):
        raise ValueError("extraction sphere crosses the outer boundary")
    x, weights = np.polynomial.legendre.leggauss(ntheta)
    # Use identical locations per hemisphere to isolate normalization from
    # angular quadrature error. On a reflected grid the two weights combine.
    x = (x+1)/2
    if not half:
        x = np.r_[-x,x]
        weights = np.r_[weights,weights]/2
    mass = angular = 0.
    omega = np.array([[0.,-1,0],[1,0,0],[0,0,0]])
    # Native y is vertical and native z is suppressed.
    pairs = ("xx","xz","xy","zz","yz","yy")
    sym = np.array([[0,1,2],[1,3,4],[2,4,5]])
    for mu,w in zip(x,weights):
        s = np.sqrt(1-mu*mu)
        rho,z = radius*s,radius*mu
        hits = np.flatnonzero((bounds[:,0]<=rho)&(rho<bounds[:,1])&
                              (bounds[:,2]<=z)&(z<bounds[:,3]))
        if len(hits)!=1:
            raise ValueError(f"ambiguous native extraction node at {(rho,z)}")
        b = hits[0]
        r0,r1,z0,z1 = bounds[b,:4]
        nz,nr = frame["mb_data"]["adm_gxx"][b].shape[-2:]
        dr,dz = (r1-r0)/(nr-1),(z1-z0)/(nz-1)
        ri,wr,dwr = interpolation((rho-r0)/dr,nr)
        zi,wz,dwz = interpolation((z-z0)/dz,nz)
        def component(prefix, axial_weights, radial_weights):
            return np.array([axial_weights @ frame["mb_data"]["adm_"+prefix+k][b][0,zi,ri]
                             @ radial_weights for k in pairs])[sym]
        g = component("g",wz,wr)
        K = component("K",wz,wr)
        grad = np.array([component("g",wz,dwr/dr), (omega@g-g@omega)/rho,
                         component("g",dwz/dz,wr)])
        n = np.array([s,0.,mu])
        mass += w*(n @ (np.einsum("jij->i",grad)-np.trace(grad,axis1=1,axis2=2)))
        # The trace-K term vanishes since the axial Killing vector is tangent.
        angular += w*rho*np.sqrt(np.linalg.det(g))*(K[1] @ np.linalg.solve(g,n))
    return dict(cycle=frame["cycle"],time=frame["time"],radius=radius,
                M_sphere=mass*radius**2/8, J_sphere=angular*radius**2/4)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot",type=Path)
    cli.add_argument("--radii",nargs="+",type=float,default=[9,12,15,17])
    a = cli.parse_args()
    frame = read_binary(a.snapshot)
    out = None
    for r in a.radii:
        row = charges(frame,r)
        if out is None:
            out = csv.DictWriter(sys.stdout,fieldnames=row)
            out.writeheader()
        out.writerow(row)
