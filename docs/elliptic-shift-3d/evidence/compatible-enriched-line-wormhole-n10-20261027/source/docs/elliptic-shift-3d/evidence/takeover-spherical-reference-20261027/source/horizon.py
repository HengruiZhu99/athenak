"""Read-only spherical horizon extraction with local polynomial interpolation."""
import numpy as np


def apparent_horizons(r,chi,expansion,width=7):
    r=np.asarray(r);chi=np.asarray(chi);expansion=np.asarray(expansion)
    weighted=np.empty_like(r);positive=r>0
    weighted[positive]=r[positive]*expansion[positive]
    weighted[~positive]=np.sqrt(chi[~positive]) # lim r*theta_plus at a regular center.
    outer=np.flatnonzero((weighted[:-1]<=0)&(weighted[1:]>0))
    allroots=np.flatnonzero((weighted[:-1]*weighted[1:]<0)|((weighted[:-1]==0)&(weighted[1:]!=0)))
    record=dict(apparent_horizon_count=len(outer),marginal_surface_crossings_total=len(allroots))
    if not len(outer):return record
    i=int(outer[-1]);fraction=-weighted[i]/(weighted[i+1]-weighted[i])
    linear_r=float(r[i]+fraction*(r[i+1]-r[i]));linear_chi=float(chi[i]+fraction*(chi[i+1]-chi[i]))
    linear_radius=linear_r/(((1-linear_r*linear_r)/2)*np.sqrt(linear_chi))
    width=min(width,len(r));start=min(max(i-(width-2)//2,0),len(r)-width)
    nodes=r[start:start+width];scale=nodes[-1]-nodes[0];x=(nodes-nodes[0])/scale
    bw=np.array([1/np.prod(x[j]-np.delete(x,j)) for j in range(width)])
    def weights(value):
        z=(value-nodes[0])/scale;delta=z-x
        if np.any(delta==0):return (delta==0).astype(float)
        v=bw/delta;return v/np.sum(v)
    left,right=float(r[i]),float(r[i+1]);method='local-polynomial'
    if np.all(np.isfinite(weighted[start:start+width])):
        for _ in range(64):
            mid=(left+right)/2
            if mid==left or mid==right:break
            value=float(weights(mid)@weighted[start:start+width])
            if value<=0:left=mid
            else:right=mid
        root=(left+right)/2;root_chi=float(weights(root)@chi[start:start+width])
    else:root=linear_r;root_chi=linear_chi;method='linear-fallback-nonfinite-stencil'
    if not np.isfinite(root_chi) or root_chi<=0:
        root=linear_r;root_chi=linear_chi;method='linear-fallback-nonpositive-interpolant'
    radius=root/(((1-root*root)/2)*np.sqrt(root_chi))
    record.update(apparent_horizon_r=float(root),apparent_horizon_areal_radius=float(radius),
        apparent_horizon_mass=float(radius/2),apparent_horizon_mass_linear=float(linear_radius/2),
        apparent_horizon_interpolation=method,apparent_horizon_interpolation_degree=width-1 if method=='local-polynomial' else 1)
    return record
