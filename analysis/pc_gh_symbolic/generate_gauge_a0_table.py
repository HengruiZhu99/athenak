#!/usr/bin/env python3
"""Generate and audit the M=1 stationary 1+log trumpet Gauge-A0 table.

The construction follows the implicit stationary 1+log solution and isotropic-radius
ODE in Bruegmann, Gen. Rel. Grav. 41, 2131 (2009), arXiv:0904.4418.  It does not use
an evolution-code implementation.  All vector/tensor quantities below are radial
scalars in isotropic Cartesian coordinates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


SQRT10 = math.sqrt(10.0)
ALPHA_C = SQRT10 - 3.0
R_C = 0.25*(SQRT10 + 3.0)
S_C = 1.0/R_C
R_ISO_C = 0.30345204271479997
C_1LOG = 0.5*R_C**3*math.exp(-ALPHA_C)
ALPHA_R_C = 8.0*(-2.0 + math.sqrt(10.0 + 3.0*SQRT10))/(16.0 + 5.0*SQRT10)


def implicit_lapse_residual(alpha: float, s: float) -> float:
    """F(alpha,S)=0, with S=1/R and M=1."""
    return alpha*alpha - 1.0 + 2.0*s - C_1LOG*math.exp(alpha)*s**4


def lapse_from_inverse_radius(s: float) -> float:
    if abs(s - S_C) < 2.0e-14:
        return ALPHA_C
    if s < S_C:
        return brentq(implicit_lapse_residual, ALPHA_C, 1.0, args=(s,),
                      xtol=5.0e-15, rtol=1.0e-14)
    return brentq(implicit_lapse_residual, 0.0, ALPHA_C, args=(s,),
                  xtol=5.0e-15, rtol=1.0e-14)


def lapse_areal_derivative(alpha: float, radius: float) -> float:
    if abs(alpha - ALPHA_C) < 2.0e-12 and abs(radius - R_C) < 2.0e-11:
        return ALPHA_R_C
    numerator = 6.0 + 4.0*radius*(alpha*alpha - 1.0)
    denominator = radius*(2.0 + radius*(alpha*alpha - 2.0*alpha - 1.0))
    return numerator/denominator


def build_table(n: int, r_min: float, r_max: float) -> dict[str, np.ndarray]:
    if n < 33 or r_min <= 0.0 or r_max <= r_min:
        raise ValueError("require n>=33 and 0<r_min<r_max")
    x = np.linspace(math.log(r_min), math.log(r_max), n)
    x_c = math.log(R_ISO_C)

    def rhs(_x: float, state: np.ndarray) -> np.ndarray:
        s = float(state[0])
        alpha = lapse_from_inverse_radius(s)
        return np.array([-s*alpha])

    common = dict(method="DOP853", rtol=2.5e-13, atol=2.5e-15, dense_output=True,
                  max_step=0.025)
    inward = solve_ivp(rhs, (x_c, float(x[0])), np.array([S_C]), **common)
    outward = solve_ivp(rhs, (x_c, float(x[-1])), np.array([S_C]), **common)
    if not inward.success or not outward.success:
        raise RuntimeError(f"trumpet ODE failed: {inward.message}; {outward.message}")
    s = np.where(x <= x_c, inward.sol(x)[0], outward.sol(x)[0])
    r = np.exp(x)
    radius = 1.0/s
    alpha = np.array([lapse_from_inverse_radius(float(value)) for value in s])
    alpha_r = np.array([
        lapse_areal_derivative(float(a), float(rr)) for a, rr in zip(alpha, radius)
    ])
    alpha_x = alpha_r*alpha*radius

    beta_norm = np.sqrt(np.maximum(0.0, alpha*alpha - 1.0 + 2.0/radius))
    beta_areal = (alpha*alpha_r - 1.0/(radius*radius))/beta_norm
    ell = beta_norm/radius
    k_radial = beta_areal
    trace_k = k_radial + 2.0*ell
    at_scalar = k_radial - ell

    lapse_sq = alpha*alpha
    lapse_sq_x = 2.0*alpha*alpha_x
    chi = (r/radius)**2
    chi_x = 2.0*chi*(1.0 - alpha)
    beta_r = r*ell
    beta_r_x = r*(ell + alpha*(beta_areal - ell))

    # Gauge A0 is obtained from the stationary target equations, not tuned.
    h_perp = (2.0 - alpha)*trace_k
    h_radial = r*(-ell*(ell + alpha*(beta_areal - ell))
                    - s*s*(alpha*alpha*(1.0 - alpha) - alpha*alpha_x))

    fields = {
        "A": lapse_sq,
        "chi": chi,
        "beta_r": beta_r,
        "K": trace_k,
        "At_radial": at_scalar,
        "h_perp": h_perp,
        "h_radial": h_radial,
    }
    table: dict[str, np.ndarray] = {"log_r": x}
    for name, values in fields.items():
        table[name] = values
        if name == "A":
            slopes = lapse_sq_x
        elif name == "chi":
            slopes = chi_x
        elif name == "beta_r":
            slopes = beta_r_x
        else:
            slopes = CubicSpline(x, values).derivative()(x)
        table[f"dx_{name}"] = slopes
    table["alpha"] = alpha
    table["R"] = radius
    return table


def audit_table(table: dict[str, np.ndarray]) -> None:
    x = table["log_r"]
    r = np.exp(x)
    alpha = table["alpha"]
    radius = table["R"]
    a = table["A"]
    chi = table["chi"]
    beta_r = table["beta_r"]
    trace_k = table["K"]

    assert np.all(np.isfinite(np.concatenate(list(table.values()))))
    assert np.all(a > 0.0) and np.all(chi > 0.0)
    assert np.all(np.diff(alpha) > 0.0)
    assert np.all(np.diff(radius) > 0.0)
    implicit = alpha*alpha - 1.0 + 2.0/radius \
               - C_1LOG*np.exp(alpha)/radius**4
    one_plus_log = beta_r*(table["dx_A"]/(2.0*alpha*r)) - 2.0*alpha*trace_k
    h_perp_definition = table["h_perp"] - (
        -alpha*trace_k + beta_r*table["dx_A"]/(2.0*a*r))
    h_radial_definition = table["h_radial"] - (
        -beta_r*table["dx_beta_r"]/r
        - 0.5*(a*table["dx_chi"] - chi*table["dx_A"])/r)
    tolerances = {
        "implicit stationary 1+log": np.max(np.abs(implicit)),
        "advective 1+log identity": np.max(np.abs(one_plus_log)),
        "h_perp target definition": np.max(np.abs(h_perp_definition)),
        "h_radial target definition": np.max(np.abs(h_radial_definition)),
    }
    for name, residual in tolerances.items():
        if residual > 2.0e-11:
            raise AssertionError(f"{name} residual {residual:.17e}")

    r0 = brentq(lambda rr: rr**4 - 2.0*rr**3 + C_1LOG, 1.0, R_C)
    gamma = (2.0 - r0)/(6.0 - 4.0*r0)
    inner = slice(0, min(64, len(x)))
    e_a = float(np.mean(np.gradient(np.log(a[inner]), x[inner])))
    e_chi = float(np.mean(np.gradient(np.log(chi[inner]), x[inner])))
    expected_a = 2.0/gamma
    if abs(e_chi - 2.0) > 3.0e-5 or abs(e_a - expected_a) > 3.0e-5:
        raise AssertionError(
            f"inner exponents e_A={e_a:.9f}, e_chi={e_chi:.9f}, "
            f"expected e_A={expected_a:.9f}, e_chi=2")
    print(f"PASS: Gauge A0 table ({len(x)} points, r=[{r[0]:.1e},{r[-1]:.1e}])")
    for name, residual in tolerances.items():
        print(f"PASS: {name}: max residual {residual:.3e}")
    print(f"PASS: inner exponents e_A={e_a:.8f}, e_chi={e_chi:.8f}")


def write_table(path: Path, table: dict[str, np.ndarray]) -> None:
    columns = ["log_r"]
    for name in ("A", "chi", "beta_r", "K", "At_radial", "h_perp", "h_radial"):
        columns.extend((name, f"dx_{name}"))
    matrix = np.column_stack([table[name] for name in columns])
    header = "PC-GH Gauge A0 M=1; derivatives are d/d(log(r/M))\n" + " ".join(columns)
    np.savetxt(path, matrix, fmt="%.17e", header=header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=4097)
    parser.add_argument("--r-min", type=float, default=1.0e-8)
    parser.add_argument("--r-max", type=float, default=1.0e4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = build_table(args.points, args.r_min, args.r_max)
    audit_table(table)
    if args.output is not None:
        write_table(args.output, table)
        print(f"WROTE: {args.output}")


if __name__ == "__main__":
    main()
