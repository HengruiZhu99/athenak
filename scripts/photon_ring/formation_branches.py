"""Offline test of the equator-symmetric MOTS ODE on native rotating ADM slices.

Usage: python formation_branches.py enclosing-09672875.json /path/to/*.bin
The first snapshot must match the archived fit. Supply subsequent slices in forward or backward order.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from numpy.polynomial import legendre as leg
from scipy.linalg import eig
from scipy.optimize import least_squares

from campaign import DIAGNOSTICS

sys.path.insert(0, str(DIAGNOSTICS.parent))
from ring_interpolation_check import LocalPolynomial
from ring_surface import NativePatch, measure, read_binary


class Curve:
    """Legendre collocation on the northern meridian, regular at both ends."""

    def __init__(self, patch, order, theta=None):
        self.patch = patch
        u, weights = leg.leggauss(3 * order)
        u = u[u > 0] if theta is None else np.cos(theta)
        self.u = u
        self.weight = np.sqrt(weights[-len(u) :]) if theta is None else np.ones_like(u)
        self.sn = sn = np.sqrt(1 - u * u)
        self.radial = np.array([sn, u, np.zeros_like(u)]).T
        self.polar = np.array([u, -sn, np.zeros_like(u)]).T
        basis = np.eye(order + 1)[:, ::2]
        self.V = leg.legval(u, basis).T
        d = leg.legval(u, leg.legder(basis)).T
        self.D = -sn[:, None] * d
        self.D2 = (
            sn[:, None] ** 2 * leg.legval(u, leg.legder(basis, 2)).T - u[:, None] * d
        )

    def geometry(self, h):
        rho, z = h * self.sn, h * self.u
        if h.min() <= 0 or max(z.max(), rho.max()) >= self.patch.bound:
            raise ValueError("surface leaves native patch")
        g = self.patch.tensor(rho, z, 0)
        inv = np.linalg.inv(g)
        K = self.patch.tensor(rho, z, 6)
        # Cartesian coordinates at phi=0 in (radial, axial, suppressed) order.
        # The suppressed derivative follows from the generator of axial rotations.
        components = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])

        def derivative(dx, dy):
            six = np.stack(
                [s.ev(z, rho, dx=dx, dy=dy) for s in self.patch.splines[:6]], axis=-1
            )
            return six[:, components]

        rot = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
        dg = np.stack(
            [
                derivative(0, 1),
                derivative(1, 0),
                (rot @ g - g @ rot) / rho[:, None, None],
            ],
            axis=1,
        )
        lowered = (dg + dg.swapaxes(1, 2) - dg.transpose(0, 2, 3, 1)) / 2
        gamma = np.einsum("nil,njkl->nijk", inv, lowered)
        return inv, K, gamma

    def expansion(self, geom, h, d, dd):
        inv, K, gamma = geom
        er, et = self.radial, self.polar
        grad = er - (d / h)[:, None] * et
        normal = np.einsum("nij,nj->ni", inv, grad)
        norm = np.sqrt((normal * grad).sum(1))
        normal = normal / norm[:, None]
        tangent = inv - normal[:, :, None] * normal[:, None, :]
        hess = (1 / h - dd / h**2)[:, None, None] * (et[:, :, None] * et[:, None, :])
        cross = er[:, :, None] * et[:, None, :]
        hess += (d / h**2)[:, None, None] * (cross + cross.swapaxes(1, 2))
        hess[:, 2, 2] = grad[:, 0] / (h * self.sn)
        hess -= np.einsum("nijk,ni->njk", gamma, grad)
        theta = np.einsum("nij,nij->n", tangent, hess / norm[:, None, None] - K)
        return theta, norm

    def operator(self, a, linearize=False):
        h, d, dd = self.V @ a, self.D @ a, self.D2 @ a
        g = self.geometry(h)
        theta, norm = self.expansion(g, h, d, dd)
        if not linearize:
            return self.weight * h * theta
        # Pointwise derivatives of the scalar ODE F(theta,h,h',h'').
        eps = self.patch.dx * 1.0e-4
        radial = (
            self.expansion(self.geometry(h + eps), h + eps, d, dd)[0]
            - self.expansion(self.geometry(h - eps), h - eps, d, dd)[0]
        ) / (2 * eps)
        slope = (
            self.expansion(g, h, d + eps, dd)[0] - self.expansion(g, h, d - eps, dd)[0]
        ) / (2 * eps)
        curvature = self.expansion(g, h, d, dd + 1)[0] - theta
        jac = (
            radial[:, None] * self.V
            + slope[:, None] * self.D
            + curvature[:, None] * self.D2
        )
        return theta, norm, jac

    def jacobian(self, a):
        theta, _, jac = self.operator(a, True)
        return self.weight[:, None] * (
            (self.V @ a)[:, None] * jac + theta[:, None] * self.V
        )

    def stability(self, a):
        # delta Theta = lambda f; a radial graph variation delta h has
        # physical normal displacement f = delta h / |grad(r-h(theta))|.
        _, norm, jac = self.operator(a, True)
        projection = self.V.T * self.weight**2
        vals, vecs = eig(projection @ jac, projection @ (self.V / norm[:, None]))
        k = np.argmin(vals.real)
        coeff = np.zeros(2 * len(a) - 1)
        coeff[::2] = vecs[:, k].real
        mode = leg.legval(np.linspace(0, 1, 1201), coeff)
        if abs(vals[k].imag) > 1.0e-6 or min(mode) * max(mode) <= 0:
            raise ValueError("principal eigenpair fails positivity or reality check")
        return vals[k].real


def fit_curve(frame, file, curve, a, order, check, interps, branch):
    patch = curve.patch
    modes = order // 2 + 1
    start = perf_counter()
    sol = least_squares(
        curve.operator, a, jac=curve.jacobian, max_nfev=45, ftol=1e-11, xtol=1e-11
    )
    elapsed = perf_counter() - start
    c = np.zeros(2 * modes - 1)
    c[::2] = sol.x
    h, plus, minus, _, _ = measure(patch, c, check)
    extrema = []
    for interp in interps:
        r, out, inc, _, _ = measure(interp, c, check)
        extrema.append([float(max(abs(r * out))), float(max(r * inc))])
    u, w = leg.leggauss(192)
    _, _, _, da, j = measure(patch, c, np.arccos(u))
    area = 2 * np.pi * np.dot(w, da / np.sqrt(1 - u * u))
    angular = 0.25 * np.dot(w, j * da / np.sqrt(1 - u * u))
    zeta = 8 * np.pi * angular / area
    residual = max(abs(h * plus))
    accepted = all(e[0] < 0.006 and e[1] < 0 for e in extrema)
    row = dict(
        file=file.name,
        time=frame["time"],
        cycle=frame["cycle"],
        seed="outer" if branch == 0 else "inner",
        order=order,
        ode_seconds=elapsed,
        nfev=sol.nfev,
        accepted=accepted,
        checks=extrema,
        residual=residual,
        max_ingoing=max(h * minus),
        min_cells=min(h) / patch.dx,
        polar_radius=h[0],
        equatorial_radius=h[len(h) // 2],
        area=area,
        J=angular,
        zeta=zeta,
        chi=2 * zeta / (1 + zeta * zeta),
    )
    if accepted:
        row["lambda0"] = curve.stability(sol.x)
        row["coeff"] = c.tolist()
    return row, c


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("snapshots", nargs="+", type=Path)
    p.add_argument("--order", type=int, default=48)
    args = p.parse_args()
    ref = json.loads(args.archive.read_text())
    outer = np.array(ref["outer"]["coeff"])[::2]
    inner = 0.93 * outer
    modes = args.order // 2 + 1
    seeds = [
        a[:modes].copy() if len(a) >= modes else np.pad(a, (0, modes - len(a)))
        for a in (outer, inner)
    ]
    check = np.r_[0.0, np.linspace(1e-5, np.pi - 1e-5, 4097), np.pi]
    for i, file in enumerate(args.snapshots):
        frame = read_binary(file)
        patch = NativePatch(frame)
        curve = Curve(patch, args.order)
        if i == 0 and frame["cycle"] != ref["cycle"]:
            raise ValueError("first slice differs from supplied onset fit")
        interps = [patch, LocalPolynomial(patch, 3), LocalPolynomial(patch, 5)]
        pair = []
        for branch, a in enumerate(seeds):
            row, c = fit_curve(
                frame, file, curve, a, args.order, check, interps, branch
            )
            if row["accepted"]:
                seeds[branch] = c[::2]
            pair.append((row, c))
        outer_fit, inner_fit = pair
        gap = leg.legval(np.cos(check), outer_fit[1] - inner_fit[1])
        mid = (outer_fit[1] + inner_fit[1]) / 2
        signs = []
        for interp in interps:
            r, out, inc, _, _ = measure(interp, mid, check)
            signs.append([float(max(r * out)), float(max(r * inc))])
        for row, _ in pair:
            separated = (
                min(gap) > 0
                and outer_fit[0].get("lambda0", 0) > 0
                and inner_fit[0].get("lambda0", 0) < 0
                and all(r[0]["accepted"] for r in pair)
            )
            row.update(
                resolved_pair=bool(separated),
                min_gap=min(gap),
                mid_surface_expansions=signs,
                strictly_trapped_mid=all(max(values) < 0 for values in signs),
            )
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    run()
