"""Seek an independently verified negative-expansion surface between a fitted MOTS pair."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from formation_branches import (
    Curve,
    LocalPolynomial,
    NativePatch,
    leg,
    measure,
    read_binary,
)


def fit_witness(frame, outer, inner, order=64):
    patch = NativePatch(frame)
    interps = [patch, LocalPolynomial(patch, 3), LocalPolynomial(patch, 5)]
    angles = np.linspace(0.0002, np.pi / 2, 321)
    audit = np.r_[0.0, np.linspace(1e-5, np.pi - 1e-5, 4097), np.pi]
    curve = Curve(patch, order, angles)
    c = np.zeros(order + 1)
    n = min(len(c), len(outer), len(inner))
    c[:n] = 0.64 * np.array(outer[:n]) + 0.36 * np.array(inner[:n])

    def expansions(coeff, theta):
        result = []
        for g in interps:
            h, plus, minus, _, _ = measure(g, coeff, theta)
            result.append((h * plus, h * minus))
        return result

    u = np.cos(audit)
    for iteration in range(12):
        checks = [
            [float(max(out)), float(max(inc))] for out, inc in expansions(c, audit)
        ]
        outer_gap = float(np.min(leg.legval(u, outer) - leg.legval(u, c)))
        inner_gap = float(np.min(leg.legval(u, c) - leg.legval(u, inner)))
        accepted = max(max(v) for v in checks) < 0 and min(outer_gap, inner_gap) > 0
        if accepted or iteration == 11:
            return dict(
                accepted=accepted,
                checks=checks,
                min_outer_gap=outer_gap,
                min_inner_gap=inner_gap,
                coeff=c.tolist(),
                iteration=iteration,
            )
        # Fix the monopole while minimizing the largest sampled outgoing expansion.
        # Use the smooth spline Jacobian as a local model for all three interpolants.
        rows = expansions(c, angles)
        upper = np.max([r[0] for r in rows], axis=0)
        jac = curve.jacobian(c[::2])[:, 1:]
        a = np.column_stack((jac, -np.ones(len(angles))))
        step = linprog(
            np.r_[np.zeros(jac.shape[1]), 1.0],
            A_ub=a,
            b_ub=-upper,
            bounds=[(-0.02, 0.02)] * jac.shape[1] + [(None, None)],
            method="highs",
        )
        if not step.success:
            raise RuntimeError(step.message)
        c[2::2] += 0.85 * step.x[:-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fits", type=Path)
    p.add_argument("frame_root", type=Path)
    args = p.parse_args()
    rows = list(map(json.loads, args.fits.read_text().splitlines()))
    for outer, inner in zip(rows[::2], rows[1::2]):
        if outer["min_gap"] < 0.01 or not all(r["accepted"] for r in (outer, inner)):
            continue
        (f,) = args.frame_root.glob("segment-*/" + outer["file"])
        result = fit_witness(read_binary(f), outer["coeff"], inner["coeff"])
        print(json.dumps(dict(file=f.name, time=outer["time"], **result)), flush=True)
