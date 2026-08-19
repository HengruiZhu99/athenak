#!/usr/bin/env python3
"""Audit binary64 stationary-trumpet source sectors at actual grid cell centers.

This uses the generated quintic-Hermite coefficient table, not the implicit
mpmath trumpet.  It mirrors the documented tetrad, spin, Cartan, and scalar
source definitions with independent NumPy point algebra, so the result can
separate table/geometry roundoff from a continuum-algebra claim.  The probe
points are the closest off-grid puncture cells for the uniform [-2,2]^3
stationary ladder.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from .covariant_gh_source_audit import (Jet, coordinate_connection,
                                             frame_covariant_source,
                                             inverse_jet_matrix)
except ImportError:
    from covariant_gh_source_audit import (Jet, coordinate_connection,
                                            frame_covariant_source,
                                            inverse_jet_matrix)


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "src/ref_gh/trumpet_table_generated.hpp"
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def table_array(text: str, name: str) -> np.ndarray:
    match = re.search(
        rf"inline constexpr Real {name}\[kTrumpetTableSize\] = \{{(.*?)\n\}};",
        text, re.DOTALL)
    if match is None:
        raise ValueError(f"missing generated table {name}")
    values = np.fromstring(match.group(1).replace(",", " "), sep=" ")
    if values.size != 4097:
        raise ValueError(f"{name} has {values.size} entries, expected 4097")
    return values


def read_table() -> tuple[dict[str, np.ndarray], float, float]:
    text = TABLE.read_text()
    coeffs = {}
    for primitive, prefix in (("alpha", "kTrumpetAlpha"),
                              ("areal", "kTrumpetArealRadius"),
                              ("q", "kTrumpetShiftQ")):
        coeffs[primitive] = np.vstack(
            [table_array(text, f"{prefix}A{power}") for power in range(6)])
    ymin = float(re.search(r"kTrumpetLogRMin = ([^;]+);", text).group(1))
    spacing = float(re.search(r"kTrumpetLogRSpacing = ([^;]+);", text).group(1))
    return coeffs, ymin, spacing


def constant(value: float) -> Jet:
    return Jet.constant(value)


def coordinate(value: float, direction: int) -> Jet:
    first = np.zeros(4)
    first[direction] = 1.0
    return Jet(value, first, np.zeros((4, 4)))


def radial_jet(value: float, derivative: float, second: float,
               point: np.ndarray, radius: float) -> Jet:
    first = np.zeros(4)
    dd = np.zeros((4, 4))
    unit = point / radius
    for i in range(3):
        first[i + 1] = derivative * unit[i]
        for j in range(3):
            dd[i + 1, j + 1] = (second * unit[i] * unit[j]
                                + derivative * ((1.0 if i == j else 0.0)
                                                - unit[i] * unit[j]) / radius)
    return Jet(value, first, dd)


def profile(coefficients: np.ndarray, ymin: float, spacing: float,
            radius: float) -> tuple[float, float, float]:
    u = (np.log(radius) - ymin) / spacing
    index = min(max(int(np.floor(u)), 0), coefficients.shape[1] - 2)
    s = u - index
    a0, a1, a2, a3, a4, a5 = coefficients[:, index]
    value = a0 + s * (a1 + s * (a2 + s * (a3 + s * (a4 + s * a5))))
    dy = (a1 + s * (2.0 * a2 + s * (3.0 * a3
          + s * (4.0 * a4 + s * 5.0 * a5)))) / spacing
    dyy = (2.0 * a2 + s * (6.0 * a3
           + s * (12.0 * a4 + s * 20.0 * a5))) / (spacing * spacing)
    return value, dy / radius, (dyy - dy) / (radius * radius)


def areal_to_psi2(areal: tuple[float, float, float], radius: float):
    value, derivative, second = areal
    inverse = 1.0 / radius
    return (value * inverse,
            derivative * inverse - value * inverse * inverse,
            second * inverse - 2.0 * derivative * inverse * inverse
            + 2.0 * value * inverse * inverse * inverse)


def exact_vacuum_riemann(areal_radius: float, point: np.ndarray) -> np.ndarray:
    """Schwarzschild R^A_BCD in the Eulerian Cartesian orthonormal frame."""
    radius = float(np.linalg.norm(point))
    normal = point / radius
    electric = (np.eye(3) - 3.0 * np.outer(normal, normal)) / areal_radius**3
    lower = np.zeros((4, 4, 4, 4))
    for i in range(3):
        for j in range(3):
            value = electric[i, j]
            lower[0, i + 1, 0, j + 1] = value
            lower[i + 1, 0, 0, j + 1] = -value
            lower[0, i + 1, j + 1, 0] = -value
            lower[i + 1, 0, j + 1, 0] = value
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ell in range(3):
                    lower[i + 1, j + 1, k + 1, ell + 1] = (
                        (1.0 if i == k else 0.0) * electric[j, ell]
                        + (1.0 if j == ell else 0.0) * electric[i, k]
                        - (1.0 if i == ell else 0.0) * electric[j, k]
                        - (1.0 if j == k else 0.0) * electric[i, ell])
    return np.einsum("a,abcd->abcd", np.diag(ETA), lower)


def reference_geometry(coefficients: dict[str, np.ndarray], ymin: float,
                       spacing: float, point: np.ndarray) -> dict:
    radius = float(np.linalg.norm(point))
    alpha = radial_jet(*profile(coefficients["alpha"], ymin, spacing, radius),
                       point, radius)
    areal = profile(coefficients["areal"], ymin, spacing, radius)
    psi2 = radial_jet(*areal_to_psi2(areal, radius), point, radius)
    shift_q = radial_jet(*profile(coefficients["q"], ymin, spacing, radius),
                         point, radius)
    inverse_alpha = alpha.reciprocal()
    inverse_psi2 = psi2.reciprocal()
    coordinates = [coordinate(point[i], i + 1) for i in range(3)]
    shift = [shift_q * item for item in coordinates]

    coframe = [[constant(0.0) for _ in range(4)] for _ in range(4)]
    coframe[0][0] = alpha
    for i in range(3):
        coframe[i + 1][0] = psi2 * shift[i]
        coframe[i + 1][i + 1] = psi2
    inverse_coframe = inverse_jet_matrix(coframe)
    frame = [[inverse_coframe[a][A] for a in range(4)] for A in range(4)]
    metric_jet = [[sum(ETA[A, A] * coframe[A][a] * coframe[A][b]
                       for A in range(4)) for b in range(4)] for a in range(4)]
    inverse_jet = [[sum(ETA[A, A] * frame[A][a] * frame[A][b]
                       for A in range(4)) for b in range(4)] for a in range(4)]
    metric, inverse, dmetric, christoffel, dchristoffel = coordinate_connection(
        metric_jet, inverse_jet)
    theta = np.array([[coframe[A][a].value for a in range(4)] for A in range(4)])
    tetrad = np.array([[frame[A][a].value for a in range(4)] for A in range(4)])
    dtheta = np.array([[[coframe[A][a].first[p] for a in range(4)]
                        for A in range(4)] for p in range(4)])
    dframe = np.array([[[frame[A][a].first[p] for a in range(4)]
                        for A in range(4)] for p in range(4)])
    ddframe = np.array([[[[frame[A][a].second[p, q] for a in range(4)]
                         for A in range(4)] for q in range(4)] for p in range(4)])

    spin = np.zeros((4, 4, 4))
    coordinate_dspin = np.zeros((4, 4, 4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for a in range(4):
                    for c in range(4):
                        covariant = dframe[c, B, a] + sum(
                            christoffel[a, c, d] * tetrad[B, d] for d in range(4))
                        spin[A, B, C] += theta[A, a] * tetrad[C, c] * covariant
                        for p in range(4):
                            d_covariant = ddframe[p, c, B, a] + sum(
                                dchristoffel[p, a, c, d] * tetrad[B, d]
                                + christoffel[a, c, d] * dframe[p, B, d]
                                for d in range(4))
                            coordinate_dspin[p, A, B, C] += (
                                (dtheta[p, A, a] * tetrad[C, c]
                                 + theta[A, a] * dframe[p, C, c]) * covariant
                                + theta[A, a] * tetrad[C, c] * d_covariant)
    for A in range(4):
        for B in range(A, 4):
            for C in range(4):
                projected = 0.5 * (ETA[A, A] * spin[A, B, C]
                                   - ETA[B, B] * spin[B, A, C])
                spin[A, B, C] = ETA[A, A] * projected
                spin[B, A, C] = -ETA[B, B] * projected
    dspin = np.einsum("Dp,pABC->DABC", tetrad, coordinate_dspin)
    for A in range(4):
        for B in range(A, 4):
            for C in range(4):
                for D in range(4):
                    projected = 0.5 * (ETA[A, A] * dspin[D, A, B, C]
                                       - ETA[B, B] * dspin[D, B, A, C])
                    dspin[D, A, B, C] = ETA[A, A] * projected
                    dspin[D, B, A, C] = -ETA[B, B] * projected
    structure = np.zeros((4, 4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(B, 4):
                structure[A, B, C] = sum(theta[A, a] * (
                    tetrad[B, p] * dframe[p, C, a]
                    - tetrad[C, p] * dframe[p, B, a])
                    for a in range(4) for p in range(4))
                structure[A, C, B] = -structure[A, B, C]
    riemann = np.zeros((4, 4, 4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for D in range(4):
                    riemann[A, B, C, D] = (dspin[C, A, B, D]
                                            - dspin[D, A, B, C])
                    for E in range(4):
                        riemann[A, B, C, D] += (
                            spin[A, E, C] * spin[E, B, D]
                            - spin[A, E, D] * spin[E, B, C]
                            - structure[E, C, D] * spin[A, B, E])
    return {"radius": radius, "alpha": alpha.value, "spin": spin,
            "dspin": dspin, "riemann": riemann,
            "analytic_riemann": exact_vacuum_riemann(areal[0], point),
            "ricci": np.einsum("abad->bd", riemann)}


def probe(resolution: int, coefficients, ymin: float, spacing: float) -> dict:
    dx = 4.0 / resolution
    point = np.full(3, 0.5 * dx)
    reference = reference_geometry(coefficients, ymin, spacing, point)
    raw_source, raw_sectors = frame_covariant_source(
        ETA, np.zeros((4, 4, 4)), reference["spin"], reference["dspin"],
        reference["riemann"], 1.0)
    source, sectors = frame_covariant_source(
        ETA, np.zeros((4, 4, 4)), reference["spin"], reference["dspin"],
        reference["analytic_riemann"], 1.0)
    maxima = {name: float(np.max(np.abs(value))) for name, value in sectors.items()
              if name in ("curvature_sector", "qq_sector", "delta_sector",
                          "damping_sector", "frame_correction")}
    return {"resolution": resolution, "dx": dx, "radius": reference["radius"],
            "source_linf": float(np.max(np.abs(source))),
            "pi_rhs_linf": float(reference["alpha"] * np.max(np.abs(source))),
            "raw_source_linf": float(np.max(np.abs(raw_source))),
            "raw_pi_rhs_linf": float(reference["alpha"] * np.max(np.abs(raw_source))),
            "frame_ricci_linf": float(np.max(np.abs(reference["ricci"]))),
            "analytic_frame_ricci_linf": float(np.max(np.abs(
                np.einsum("abad->bd", reference["analytic_riemann"])))),
            "raw_riemann_minus_analytic_linf": float(np.max(np.abs(
                reference["riemann"] - reference["analytic_riemann"]))),
            "q_linf": float(np.max(np.abs(sectors["q"]))),
            "delta_linf": float(max(np.max(np.abs(sectors["delta_lower"])),
                                     np.max(np.abs(sectors["delta"])))),
            "sectors": maxima}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    coefficients, ymin, spacing = read_table()
    result = {"table": str(TABLE.relative_to(ROOT)),
              "probes": [probe(n, coefficients, ymin, spacing)
                         for n in args.resolutions]}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
