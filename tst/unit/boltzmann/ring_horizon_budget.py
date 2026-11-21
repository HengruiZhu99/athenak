"""Check the axial angular-momentum balance and native constraints on a fitted surface."""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre as leg

from ring_surface import NativePatch, measure, read_binary
from ring_reflection import reflected_equator


def native_values(frame, rho, z, key):
    values = np.full(len(rho), np.nan)
    spacing = np.full(len(rho), np.inf)
    for block, domain in enumerate(frame["mb_geometry"]):
        data = frame["mb_data"][key][block][0]
        nz, nr = data.shape
        dx = (domain[1] - domain[0])/(nr-1)
        sel = ((rho >= domain[0]) & (rho <= domain[1]) & (z >= domain[2])
               & (z <= domain[3]) & (dx < spacing))
        if not np.any(sel):
            continue
        x = np.clip((rho[sel]-domain[0])/dx, 0, nr-1)
        y = np.clip((z[sel]-domain[2])*(nz-1)/(domain[3]-domain[2]), 0, nz-1)
        i, j = np.minimum(x.astype(int), nr-2), np.minimum(y.astype(int), nz-2)
        a, b = x-i, y-j
        values[sel] = ((1-a)*(1-b)*data[j,i] + a*(1-b)*data[j,i+1]
                       + (1-a)*b*data[j+1,i] + a*b*data[j+1,i+1])
        spacing[sel] = dx
    if not np.all(np.isfinite(values)):
        raise ValueError(f"uncovered or invalid surface sample for {key}")
    return values


def budget(snapshot, fit, mass, initial_j):
    adm = read_binary(snapshot)
    field = read_binary(snapshot.with_name(snapshot.name.replace(".adm.", ".tmunu.")))
    np.testing.assert_array_equal(adm["mb_geometry"], field["mb_geometry"])
    if adm["cycle"] != field["cycle"] or not np.isclose(adm["time"],fit["time"]):
        raise ValueError("snapshot and candidate times differ")
    coeff = np.array(fit["coeff"])
    signs = (1., -1.) if reflected_equator(field) else (1.,)
    center_z = fit.get("center_z", 0.)
    full, exterior = 0., 0.
    q, w = leg.leggauss(4)
    for block, domain in enumerate(field["mb_geometry"]):
        sphi = field["mb_data"]["tmunu_Sz"][block][0]
        nz, nr = sphi.shape
        dr, dz = (domain[1]-domain[0])/nr, (domain[3]-domain[2])/nz
        rho, z = np.meshgrid(domain[0]+(np.arange(nr)+.5)*dr, domain[2]+(np.arange(nz)+.5)*dz)
        g=[]
        for suffix in ("xx","xy","xz","yy","yz","zz"):
            v=adm["mb_data"]["adm_g"+suffix][block][0].astype(float)
            if v.shape != (nz+1,nr+1):
                raise ValueError("axial budget expects vertex-centered ADM geometry")
            g.append(.25*(v[:-1,:-1]+v[1:,:-1]+v[:-1,1:]+v[1:,1:]))
        xx,xy,xz,yy,yz,zz = g
        det=xx*yy*zz+2*xy*xz*yz-xx*yz**2-yy*xz**2-zz*xy**2
        if np.any(det<=0):
            raise ValueError("nonpositive metric determinant in volume integral")
        dj = 2*np.pi*dr*dz*np.sqrt(det)*rho**2*sphi
        full += len(signs)*dj.sum()
        fraction = np.zeros_like(rho)
        for u,wu in zip(q,w):
            for v,wv in zip(q,w):
                for sign in signs:
                    local_z = sign*(z+.5*dz*v)-center_z
                    r = np.hypot(rho+.5*dr*u, local_z)
                    fraction += .25*wu*wv*(r > leg.legval(local_z/r, coeff))
        exterior += (fraction*dj).sum()
    j = fit.get("surface_J",fit.get("JH"))
    result=dict(time=fit["time"],cycle=int(adm["cycle"]),surface_J=j,matter_J=full,
                exterior_J=exterior,JH_plus_Jext_minus_JADM=(j+exterior-initial_j),
                JADM=initial_j)
    path = snapshot.with_name(snapshot.name.replace(".adm.",".con."))
    if path.exists():
        frame = read_binary(path)
        if adm["cycle"] != frame["cycle"]:
            raise ValueError("constraint and surface times differ")
        x,w = leg.leggauss(160)
        theta=np.arccos(x)
        h,_,_,da,_=measure(NativePatch(adm),coeff,theta,center_z)
        darea=w*da/np.sin(theta)
        rho,z=h*np.sin(theta),h*x+center_z
        for key,label in [("con_H","H"),("con_M","M")]:
            sample_z = abs(z) if reflected_equator(frame) else z
            value = native_values(frame,rho,sample_z,key)
            if label=="M":
                value=np.sqrt(np.maximum(value,0))
            result[label+"_rms_M2"]=mass**2*np.sqrt(darea @ (value*value)/darea.sum())
            result[label+"_max_M2"]=mass**2*np.max(abs(value))
    return result


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot",type=Path)
    cli.add_argument("fits",type=Path,help="JSON lines from ring_surface.py on the same snapshot")
    cli.add_argument("--mass",type=float,required=True)
    cli.add_argument("--J-adm",type=float,required=True)
    args=cli.parse_args()
    for line in args.fits.read_text().splitlines():
        print(json.dumps(budget(args.snapshot,json.loads(line),args.mass,args.J_adm)))
