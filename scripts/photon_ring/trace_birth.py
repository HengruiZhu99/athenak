"""Continue a centered MOTS pair through a fold on interpolated ADM geometry."""

import copy

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import brentq, root
from formation_branches import Curve, NativePatch, LocalPolynomial, leg, measure


CHECK = np.r_[0.0, np.linspace(1.0e-5, np.pi - 1.0e-5, 4097), np.pi]


class History:
    def __init__(self, frames, mass, stride, method):
        self.times = np.array([f["time"] / mass for f in frames])
        patches = [NativePatch(f, stride) for f in frames]
        self.base = patches[0]
        # Interpolate the spatial spline coefficients: these are linear functions
        # of the ADM data on the same knots and commute with time interpolation.
        for p in patches:
            assert p.dx == self.base.dx and p.bound == self.base.bound
            for s, b in zip(p.splines, self.base.splines):
                assert all(
                    np.array_equal(v, w) for v, w in zip(s.get_knots(), b.get_knots())
                )
        self.coeff = np.array([[s.get_coeffs() for s in p.splines] for p in patches])
        self.lo, self.hi = self.times[[1, 2]]
        deriv = (self.coeff[1::2] - self.coeff[::2]) / np.diff(self.times)[
            ::2, None, None
        ]
        self.hermite = (
            CubicSpline(self.times, self.coeff)
            if method == "four-slice cubic"
            else CubicHermiteSpline([self.lo, self.hi], self.coeff[[1, 2]], deriv)
        )

    def geometry(self, t, method):
        f = (t - self.lo) / (self.hi - self.lo)
        co = (
            (1 - f) * self.coeff[1] + f * self.coeff[2]
            if method == "linear"
            else self.hermite(t)
        )
        return self.from_coefficients(co)

    def from_coefficients(self, co):
        patch = copy.copy(self.base)
        patch.splines = []
        for s, c in zip(self.base.splines, co):
            sp = copy.copy(s)
            sp.tck = (*s.get_knots(), c)
            patch.splines.append(sp)
        return patch


def trace(history, order, method, seed_inner, seed_outer, mass):
    curve = Curve(history.base, order)
    cache = {}

    def solve_fixed(mean):
        if mean in cache:
            return cache[mean]
        if cache:
            previous = cache[min(cache, key=lambda k: abs(k - mean))]
            y = np.r_[previous["t_M"], previous["coeff"][2::2]]
        else:
            y = np.r_[history.hi, seed_inner[2 : order + 1 : 2]]
        project = curve.V.T * curve.weight

        def fun(x):
            curve.patch = history.geometry(x[0], method)
            return project @ curve.operator(np.r_[mean, x[1:]])

        def jac(x):
            a = np.r_[mean, x[1:]]
            e = 0.0001
            dt = (fun(x + np.r_[e, 0 * x[1:]]) - fun(x - np.r_[e, 0 * x[1:]])) / (2 * e)
            curve.patch = history.geometry(x[0], method)
            return np.column_stack([dt, project @ curve.jacobian(a)[:, 1:]])

        sol = root(fun, y, jac=jac, tol=2.0e-10)
        if np.max(np.abs(fun(sol.x))) > 1.0e-8:
            raise RuntimeError(sol.message)
        t = sol.x[0]
        if t < history.lo - 0.01 or t > history.hi + 0.03:
            raise ValueError(f"outside interpolation window: {t}")
        a = np.r_[mean, sol.x[1:]]
        c = np.zeros(order + 1)
        c[::2] = a
        lam = curve.stability(a) * mass**2
        u, w = leg.leggauss(160)
        _, _, _, da, j = measure(curve.patch, c, np.arccos(u))
        area = 2 * np.pi * (w @ (da / np.sqrt(1 - u * u)))
        angular = 0.25 * (w @ (da * j / np.sqrt(1 - u * u)))
        zeta = 8 * np.pi * angular / area
        r, p, m, _, _ = measure(curve.patch, c, CHECK)
        rec = dict(
            t_M=t,
            mean=mean,
            polar=r[0],
            equator=r[len(r) // 2],
            area=area,
            J=angular,
            chi=2 * zeta / (1 + zeta * zeta),
            zeta=zeta,
            lambda_M2=lam,
            residual=max(abs(r * p)),
            max_in=max(r * m),
            coeff=c.tolist(),
        )
        cache[mean] = rec
        return rec

    span = seed_outer[0] - seed_inner[0]
    means = np.linspace(seed_inner[0] + 0.09 * span, seed_outer[0] - 0.09 * span, 21)
    points = [solve_fixed(v) for v in means]
    lo, hi = next(
        (x["mean"], y["mean"])
        for x, y in zip(points, points[1:])
        if x["lambda_M2"] < 0 < y["lambda_M2"]
    )
    q = brentq(lambda a: solve_fixed(a)["lambda_M2"], lo, hi, xtol=1e-9)
    fold = dict(solve_fixed(q))
    eps = 0.0005
    ta, tb = [solve_fixed(q + v)["t_M"] for v in [-eps, eps]]
    fold.update(
        dt_dmean=(tb - ta) / (2 * eps), d2t_dmean2=(tb + ta - 2 * fold["t_M"]) / eps**2
    )
    interp = history.geometry(fold["t_M"], method)
    extrema = []
    for patch in [interp, LocalPolynomial(interp, 3), LocalPolynomial(interp, 5)]:
        h, p, m, _, _ = measure(patch, fold["coeff"], CHECK)
        extrema.append([max(abs(h * p)), max(h * m)])
    if (
        max(v[0] for v in extrema) > 0.006
        or max(v[1] for v in extrema) >= 0
        or fold["d2t_dmean2"] <= 0
    ):
        raise ValueError(
            f"unqualified reconstructed fold: {extrema}, {fold['d2t_dmean2']}"
        )
    fold["expansion_checks"] = extrema
    # Independent of the spin functional, test the separation of the two radii
    # at equal times after the minimum. Use only the local monotone interval.
    edges = [min(points, key=lambda p: p["mean"]), max(points, key=lambda p: p["mean"])]
    interval = min(p["t_M"] - fold["t_M"] for p in edges)
    if interval > 1.0e-5:
        paired = []
        for dt in interval * np.array([0.01, 0.02, 0.04, 0.08, 0.16]):

            def at_time(mean):
                return solve_fixed(mean)["t_M"] - fold["t_M"] - dt

            left = solve_fixed(brentq(at_time, edges[0]["mean"], q))
            right = solve_fixed(brentq(at_time, q, edges[1]["mean"]))
            paired.append(
                dict(
                    dt_M=float(dt),
                    polar_gap=right["polar"] - left["polar"],
                    equatorial_gap=right["equator"] - left["equator"],
                )
            )
        fold["paired_radii"] = paired
        fold["fitted_exponents"] = {
            key: float(
                np.polyfit(
                    np.log([p["dt_M"] for p in paired]),
                    np.log([p[key] for p in paired]),
                    1,
                )[0]
            )
            for key in ("polar_gap", "equatorial_gap")
            if all(p[key] > 0 for p in paired)
        }
    return fold, points
