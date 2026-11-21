"""Independent axisymmetric expansion and horizon-spin measurement on ADM snapshots."""

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import legendre as leg
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vis/python"))
from bin_convert import read_binary


class NativePatch:
    """A contiguous uniform patch of the finest native nodes surrounding the origin."""

    def __init__(self, frame):
        names = ["adm_" + prefix + suffix for prefix in ("g", "K")
                 for suffix in ("xx", "xy", "xz", "yy", "yz", "zz")]
        values = frame["mb_data"]
        patches = []
        for b, bounds in enumerate(frame["mb_geometry"]):
            nz, nr = values[names[0]][b][0].shape
            dr = (bounds[1] - bounds[0]) / (nr - 1)
            dz = (bounds[3] - bounds[2]) / (nz - 1)
            if not np.isclose(dr, dz):
                raise ValueError("surface analysis expects square native cells")
            patches.append((dr, bounds, b, nr, nz))
        self.dx = h = min(p[0] for p in patches)
        patches = [p for p in patches if np.isclose(p[0], h)]
        high = round(max(p[1][1] for p in patches) / h)
        low_z = round(min(p[1][2] for p in patches) / h)
        high_z = round(max(p[1][3] for p in patches) / h)
        grid = np.full((12, high_z-low_z+1, high+1), np.nan)
        for _, bounds, b, nr, nz in patches:
            i, j = round(bounds[0]/h), round(bounds[2]/h)-low_z
            grid[:, j:j+nz, i:i+nr] = [values[k][b][0] for k in names]
        # Mirror only a declared equatorial boundary. A finest patch wholly
        # north of the plane in a full-domain file is a different situation.
        mesh = re.search(r"(?ms)^<mesh>\n(.*?)(?=^<|\Z)", "\n".join(frame["header"]))
        if mesh and re.search(r"(?m)^ix2_bc\s*=\s*reflect\s*$", mesh[1]):
            if low_z != 0:
                raise ValueError("the finest equatorial patch must touch z=0")
            parity = np.array([1, -1, 1, 1, -1, 1] * 2)
            grid = np.concatenate((parity[:, None, None]*grid[:, :0:-1, :], grid), axis=1)
            low_z = -high_z
        extent = min(high, -low_z, high_z)
        while extent > 3:
            region = grid[:, -low_z-extent:-low_z+extent+1, :extent+1]
            if np.isfinite(region).all():
                break
            extent -= 1
        if extent <= 3:
            raise ValueError("the finest grid lacks a resolved central patch")
        self.bound = extent * h
        # In native (radial, axial, suppressed) Cartesian components a half-turn
        # flips both transverse indices. Extend smoothly across signed radius.
        parity = np.array([1, -1, 1, 1, -1, 1] * 2)
        full = np.concatenate((parity[:, None, None] * region[:, :, :0:-1], region), axis=2)
        coord = np.arange(-extent, extent+1) * h
        self.splines = [RectBivariateSpline(coord, coord, v, s=0) for v in full]

    def tensor(self, rho, z, start):
        six = np.stack([f.ev(z, rho) for f in self.splines[start:start+6]], axis=-1)
        return six[:, [[0, 1, 2], [1, 3, 4], [2, 4, 5]]]


def measure(patch, coeff, theta):
    """Compute D_i s^i through the cylindrical divergence of the unit normal."""
    poles = (theta == 0) | (theta == np.pi)
    u, sn = np.cos(theta), np.where(poles, 0., np.sin(theta))
    h = leg.legval(u, coeff)
    dh = -sn * leg.legval(u, leg.legder(coeff))
    rho, z = h*sn, h*u
    step = 1e-3 * patch.dx
    if h.min() <= 0 or max(abs(rho).max(), abs(z).max()) + step >= patch.bound:
        raise ValueError("trial surface outside native patch")

    def normal(r, y):
        radius = np.hypot(r, y)
        sine, cosine = r/radius, y/radius
        derivative = -sine * leg.legval(cosine, leg.legder(coeff))
        dF = np.stack((sine - derivative*cosine/radius,
                       cosine + derivative*sine/radius, np.zeros_like(r)), axis=-1)
        g = patch.tensor(r, y, 0)
        determinant = np.linalg.det(g)
        minor = g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1]**2
        if not np.all((g[:, 0, 0] > 0) & (minor > 0) & (determinant > 0)):
            raise ValueError("interpolated spatial metric is not positive definite")
        inv = np.linalg.inv(g)
        upper = np.einsum("nij,nj->ni", inv, dF)
        n = upper / np.sqrt(np.sum(upper*dF, axis=1))[:, None]
        volume = r * np.sqrt(determinant)
        return n * volume[:, None], volume, n, g, inv

    _, volume, n, g, inv = normal(rho, z)
    radial = normal(rho+step, z)[0][:, 0]-normal(rho-step, z)[0][:, 0]
    axial = normal(rho, z+step)[0][:, 1]-normal(rho, z-step)[0][:, 1]
    div = np.divide(radial+axial, 2*step*volume, out=np.zeros_like(volume), where=~poles)
    if np.any(poles):
        # At the regular axis the cylindrical radial divergence tends to
        # twice the derivative of sqrt(gamma) s^rho. Evaluate in signed rho.
        zp = z[poles]
        zero = np.zeros_like(zp)
        def densitized(r, y, component):
            _, _, unit, metric, _ = normal(r, y)
            return np.sqrt(np.linalg.det(metric))*unit[:,component]
        dr = (densitized(zero+step,zp,0) - densitized(zero-step,zp,0))/(2*step)
        dz = (densitized(zero,zp+step,1) - densitized(zero,zp-step,1))/(2*step)
        div[poles] = (2*dr+dz)/np.sqrt(np.linalg.det(g[poles]))
    k = patch.tensor(rho, z, 6)
    projected = np.einsum("ni,nij,nj->n", n, k, n) - np.einsum("nij,nij->n", inv, k)
    tangent = np.stack((dh*sn+h*u, dh*u-h*sn, np.zeros_like(h)), axis=-1)
    t2 = np.einsum("ni,nij,nj->n", tangent, g, tangent)
    cross = np.einsum("ni,ni->n", tangent, g[:, :, 2])
    da = rho*np.sqrt(t2*g[:, 2, 2]-cross**2)
    j = rho*np.sum(k[:, 2, :]*n, axis=1)
    return h, projected+div, projected-div, da, j


def solve_surface(patch, seed, target, order, max_evals, initial_coeff=None):
    x, w = leg.leggauss(max(32, 3*order))
    theta = np.arccos(x)

    def unpack(a):
        coeff = np.zeros(2*len(a)-1)
        coeff[::2] = a
        return coeff

    def residual(a):
        try:
            h, out, _, _, _ = measure(patch, unpack(a), theta)
        except (ValueError, np.linalg.LinAlgError):
            return np.full_like(w, 100.0)
        return np.sqrt(w)*(h*out-target)

    a = np.zeros(order//2+1)
    a[0] = seed
    if initial_coeff is not None:
        previous = np.asarray(initial_coeff)[::2]
        count = min(len(previous), len(a))
        a[:count] = previous[:count]
    fit = least_squares(residual, a, max_nfev=max_evals, ftol=1e-10, xtol=1e-10)
    c = unpack(fit.x)
    h, out, ingoing, da, j = measure(patch, c, theta)
    area = 2*np.pi * (w @ (da/np.sin(theta)))
    angular = 0.25 * (w @ (j*da/np.sin(theta)))
    mirr2 = area / (16*np.pi)
    mass2 = mirr2 + angular**2 / (4*mirr2)
    # Include the poles' regular limiting neighborhoods and independent angles.
    fine = np.r_[0., np.linspace(1e-4, np.pi-1e-4, 257), np.pi]
    r, plus, minus, _, _ = measure(patch, c, fine)
    return dict(target=target, seed=seed, h_min=float(r.min()), h_max=float(r.max()),
                zones=float(r.min()/patch.dx), max_rTheta=float(max(r*plus)),
                min_rTheta=float(min(r*plus)), max_rThetaMinus=float(max(r*minus)),
                sampled_future_trapped=bool(np.all(plus < 0) and np.all(minus < 0)),
                area=float(area), surface_mass=float(np.sqrt(mass2)),
                surface_J=float(angular), surface_spin_ratio=float(angular/mass2),
                area_spin_gap=float(1 - 8*np.pi*abs(angular)/area), coeff=c.tolist())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--seeds", type=float, nargs="+", required=True)
    cli.add_argument("--target", type=float, default=0.0)
    cli.add_argument("--order", type=int, default=8)
    cli.add_argument("--max-evals", type=int, default=80)
    args = cli.parse_args()
    frame = read_binary(args.snapshot)
    patch = NativePatch(frame)
    usable = 0
    for seed in args.seeds:
        try:
            row = solve_surface(patch, seed, args.target, args.order, args.max_evals)
        except (ValueError, np.linalg.LinAlgError) as error:
            print(f"Rejected seed {seed:g}: {error}", file=sys.stderr)
            continue
        print(json.dumps(dict(time=frame["time"], dx=patch.dx, **row), allow_nan=False), flush=True)
        usable += 1
    if not usable:
        raise SystemExit("No usable surface fits")
