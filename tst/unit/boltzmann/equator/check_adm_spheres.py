"""Check finite-sphere charge normalization on isotropic and Bowen–York exterior fields."""
import numpy as np
from adm_spheres import charges


def exterior(spacing, half):
    r = np.arange(0,8+spacing/2,spacing)
    z = np.arange(0 if half else -8,8+spacing/2,spacing)
    x,y = np.meshgrid(r,z)
    dist = np.maximum(np.hypot(x,y),.1)
    # Units M=1, J=0.93. Only the smooth exterior enters the measurement.
    psi = 1+.5/dist
    n = np.stack([x/dist,y/dist,np.zeros_like(x)])
    tangent = np.stack([np.zeros_like(x),np.zeros_like(x),.93*x/dist])
    values = {}
    names = "xyz"
    for i in range(3):
        for j in range(i,3):
            ij = names[i]+names[j]
            values["adm_g"+ij] = [((i==j)*psi**4)[None]]
            K = 3*(tangent[i]*n[j]+n[i]*tangent[j])/(psi**2*dist**3)
            values["adm_K"+ij] = [K[None]]
    if half:
        bounds = [[0,8,0,8,0,1]]
    else:
        mid = len(z)//2
        values = {key:[a[0][:,:mid+1,:],a[0][:,mid:,:]] for key,a in values.items()}
        bounds = [[0,8,-8,0,0,1],[0,8,0,8,0,1]]
    return dict(mb_geometry=bounds,mb_data=values,cycle=0,time=0,
                header=["<mesh>",f"ix2_bc = {'reflect' if half else 'outflow'}"])


if __name__ == "__main__":
    radius = 5.
    exact = np.array([(1+.5/radius)**3,.93])
    for spacing in (.25,.125,.0625):
        pairs = []
        for half in (False,True):
            v = charges(exterior(spacing,half),radius)
            pairs.append(np.array([v["M_sphere"],v["J_sphere"]]))
        np.testing.assert_allclose(*pairs, atol=2.e-13,rtol=0)
        err = np.abs(pairs[0]-exact)
        if spacing==.0625:
            np.testing.assert_allclose(pairs[0],exact,atol=1.e-6,rtol=0)
        print(spacing, pairs[0], pairs[1], err)
