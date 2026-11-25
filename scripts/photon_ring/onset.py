"""Reduce saved geometry to separately qualified two-branch and strict-trapping results."""

import numpy as np
from formation_branches import (
    Curve,
    NativePatch,
    LocalPolynomial,
    fit_curve,
    leg,
    read_binary,
)

CHECK = np.r_[0.0, np.linspace(1e-5, np.pi - 1e-5, 4097), np.pi]


def pair(file, seeds, order=48, stride=1):
    frame = read_binary(file)
    patch = NativePatch(frame, stride)
    curve = Curve(patch, order)
    interps = [patch, LocalPolynomial(patch, 3), LocalPolynomial(patch, 5)]
    candidates = []
    for label, seed, factor in [
        (0, seeds[0], 1.0),
        (1, seeds[1], 1.0),
        (0, seeds[0], 1.08),
        (1, seeds[0], 0.92),
        (1, seeds[0], 0.82),
        (0, seeds[0], 1.2),
    ]:
        a = np.zeros(order // 2 + 1)
        coeff = np.asarray(seed)[::2]
        n = min(len(a), len(coeff))
        a[:n] = factor * coeff[:n]
        try:
            row, _ = fit_curve(frame, file, curve, a, order, CHECK, interps, label)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not row["accepted"]:
            continue
        candidates.append(row)
        for outer in candidates:
            for inner in candidates:
                if outer["lambda0"] <= 0 or inner["lambda0"] >= 0:
                    continue
                gap = leg.legval(
                    np.cos(CHECK), np.asarray(outer["coeff"]) - inner["coeff"]
                )
                if min(gap) <= 0:
                    continue
                outer["seed"], inner["seed"] = "outer", "inner"
                return dict(
                    cycle=outer["cycle"],
                    time=outer["time"],
                    outer=outer,
                    inner=inner,
                    min_gap=float(min(gap)),
                )
    return None


def qualify(file, seeds, mass):
    p = pair(file, seeds)
    if p is None:
        return None
    other = pair(file, [p["outer"]["coeff"], p["inner"]["coeff"]], 40)
    if other is None:
        return None
    p["order40"] = dict(chi=other["outer"]["chi"], min_gap=other["min_gap"])
    # Demand resolvable separation under change of fit order.
    differences = [
        np.max(
            abs(leg.legval(np.cos(CHECK), leg.legsub(p[k]["coeff"], other[k]["coeff"])))
        )
        for k in ["outer", "inner"]
    ]
    if p["min_gap"] <= sum(differences):
        return None
    p["t_M"] = p["time"] / mass
    return p


def fold_fit(points):
    """Estimate a common birth time from two squared branch separations.
    Require three distinct epochs and positive slopes. Window spread is reported.
    """
    if len(points) < 4:
        return None
    points = sorted(points, key=lambda p: p["t_M"])
    time = np.array([p["t_M"] for p in points])
    records = []
    for n in sorted(set([max(3, len(points) // 2), len(points)])):
        for key in ["polar_radius", "equatorial_radius"]:
            radii = np.array([p["outer"][key] - p["inner"][key] for p in points[:n]])
            slope, offset = np.polyfit(time[:n] - time[0], radii**2, 1)
            if slope <= 0:
                continue
            t0 = time[0] - offset / slope
            if t0 > time[0] or time[0] - t0 > 2 * (time[n - 1] - time[0]):
                continue

            # Common drift of J and A is regular across the fold.
            def at_birth(key):
                avg = np.array(
                    [0.5 * (p["outer"][key] + p["inner"][key]) for p in points[:n]]
                )
                return np.polyval(np.polyfit(time[:n] - time[0], avg, 1), t0 - time[0])

            z = 8 * np.pi * at_birth("J") / at_birth("area")
            records.append(
                dict(epochs=n, coordinate=key, t_M=t0, zeta=z, chi=2 * z / (1 + z * z))
            )
    return records or None
