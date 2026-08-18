#!/usr/bin/env python3
"""Generate the unit-mass, n=2 stationary 1+log trumpet table.

This follows the physical-root and split-integral construction used by
SpECTRE's TrumpetSchwarzschild solution.  The generated table is uniform in
logarithmic isotropic radius y=ln(r/M).  It stores the regular primitives
alpha(y), R(y)/M, and M q(y), together with their first and second y
derivatives for one device-side piecewise-quintic Hermite interpolant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq


N = 2.0
ALPHA_SPLIT = 0.1


def constants() -> tuple[float, float, float]:
    alpha_c = np.sqrt((np.sqrt(4.0 + 9.0 * N**2) - 3.0 * N)
                      / (np.sqrt(4.0 + 9.0 * N**2) + 3.0 * N))
    radius_c = (3.0 * N**2 + np.sqrt(4.0 * N**2 + 9.0 * N**4)) / (4.0 * N**2)
    c_squared = ((3.0 * N + np.sqrt(4.0 + 9.0 * N**2))**3
                 / (128.0 * N**3) * np.exp(-2.0 * alpha_c / N))
    return alpha_c, radius_c, c_squared


ALPHA_C, RADIUS_C, C_SQUARED = constants()


def critical_physical_slope() -> float:
    # At the critical double root F=F_R=0.  Twice differentiating
    # F(alpha,R(alpha))=0 gives a quadratic for dR/dalpha.  The positive root
    # is the physical branch that grows toward spatial infinity.
    f_rr = 12.0 * (1.0 - ALPHA_C**2) * RADIUS_C**2 - 12.0 * RADIUS_C
    twice_f_ra = -16.0 * ALPHA_C * RADIUS_C**3
    f_aa = (-2.0 * RADIUS_C**4
            + 4.0 / N**2 * C_SQUARED * np.exp(2.0 * ALPHA_C / N))
    roots = np.roots([f_rr, twice_f_ra, f_aa])
    return float(max(roots))


CRITICAL_PHYSICAL_SLOPE = critical_physical_slope()


def critical_physical_curvature() -> float:
    """Return d2R/dalpha2 on the physical branch at the critical point."""
    a = ALPHA_C
    radius = RADIUS_C
    p = CRITICAL_PHYSICAL_SLOPE
    exponential = C_SQUARED * np.exp(2.0 * a / N)
    f_rr = 12.0 * (1.0 - a**2) * radius**2 - 12.0 * radius
    f_ra = -8.0 * a * radius**3
    f_rrr = 24.0 * (1.0 - a**2) * radius - 12.0
    f_rra = -24.0 * a * radius**2
    f_raa = -8.0 * radius**3
    f_aaa = 8.0 / N**3 * exponential
    numerator = f_rrr * p**3 + 3.0 * f_rra * p**2 + 3.0 * f_raa * p + f_aaa
    return -numerator / (3.0 * (f_rr * p + f_ra))


CRITICAL_PHYSICAL_CURVATURE = critical_physical_curvature()


def implicit(alpha: float | np.ndarray,
             radius: float | np.ndarray) -> float | np.ndarray:
    return ((1.0 - alpha * alpha) * radius**4 - 2.0 * radius**3
            + C_SQUARED * np.exp(2.0 * alpha / N))


def radius_from_alpha(alpha: float) -> float:
    """Return the physical areal-radius root, including the double-root limit."""
    if abs(alpha - ALPHA_C) < 2.0e-13:
        return RADIUS_C
    if alpha < ALPHA_C:
        lower, upper = 0.0, RADIUS_C
    else:
        lower = RADIUS_C
        upper = max(5000.0, 4.0 / (1.0 - alpha * alpha))
    f_lower = implicit(alpha, lower)
    f_upper = implicit(alpha, upper)
    if f_lower == 0.0:
        return lower
    if f_upper == 0.0:
        return upper
    if f_lower * f_upper > 0.0:
        # Roundoff can lift the critical double root by a few ulps.
        scale = max(1.0, abs(C_SQUARED * np.exp(2.0 * alpha / N)))
        if abs(alpha - ALPHA_C) < 2.0e-6 and abs(f_lower) < 2.0e-11 * scale:
            delta = alpha - ALPHA_C
            return (RADIUS_C + CRITICAL_PHYSICAL_SLOPE * delta
                    + 0.5 * CRITICAL_PHYSICAL_CURVATURE * delta**2)
        raise RuntimeError(
            f"physical-root bracket failed at alpha={alpha:.17e}: "
            f"f(lower)={f_lower:.17e}, f(upper)={f_upper:.17e}"
        )
    return brentq(lambda radius: implicit(alpha, radius), lower, upper,
                  xtol=1.0e-14, rtol=4.0 * np.finfo(float).eps)


def d_alpha_d_radius(alpha: float | np.ndarray,
                      radius: float | np.ndarray) -> float | np.ndarray:
    return N * (2.0 * radius - 3.0 - 2.0 * radius * alpha**2) / (
        radius * (radius - 2.0 + N * radius * alpha - radius * alpha**2)
    )


def alpha_radius_derivatives(alpha: np.ndarray,
                             radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return d(alpha)/dR and d2(alpha)/dR2 with critical regularization."""
    first = np.asarray(d_alpha_d_radius(alpha, radius)).copy()
    second = np.empty_like(first)
    near = np.abs(alpha - ALPHA_C) < 2.0e-7
    delta = alpha[near] - ALPHA_C
    radius_alpha = CRITICAL_PHYSICAL_SLOPE + CRITICAL_PHYSICAL_CURVATURE * delta
    first[near] = 1.0 / radius_alpha
    second[near] = -CRITICAL_PHYSICAL_CURVATURE / radius_alpha**3

    far = ~near
    if np.any(far):
        a = alpha[far]
        r = radius[far]
        exponential = C_SQUARED * np.exp(2.0 * a / N)
        f_a = -2.0 * a * r**4 + 2.0 / N * exponential
        f_rr = 12.0 * (1.0 - a**2) * r**2 - 12.0 * r
        f_ra = -8.0 * a * r**3
        f_aa = -2.0 * r**4 + 4.0 / N**2 * exponential
        second[far] = -(f_rr + 2.0 * f_ra * first[far]
                        + f_aa * first[far]**2) / f_a
    return first, second


def alpha_from_radius(radius: float) -> float:
    """Invert the physical branch as alpha(R), which is regular at R_c."""
    if radius <= radius_from_alpha(0.0):
        return 0.0
    if abs(radius - RADIUS_C) < 2.0e-6:
        delta_radius = radius - RADIUS_C
        delta_alpha = (2.0 * delta_radius
                       / (CRITICAL_PHYSICAL_SLOPE + np.sqrt(
                           CRITICAL_PHYSICAL_SLOPE**2
                           + 2.0 * CRITICAL_PHYSICAL_CURVATURE * delta_radius)))
        return ALPHA_C + delta_alpha
    lower, upper = ((0.0, ALPHA_C) if radius < RADIUS_C
                    else (ALPHA_C, 1.0 - 1.0e-14))
    return brentq(lambda alpha: implicit(alpha, radius), lower, upper,
                  xtol=1.0e-14,
                  rtol=4.0 * np.finfo(float).eps)


def first_integral_above(alpha: float) -> float:
    points = [ALPHA_C] if min(ALPHA_SPLIT, alpha) < ALPHA_C < max(ALPHA_SPLIT, alpha) else None
    return quad(lambda value: np.log(radius_from_alpha(value)) / value**2,
                ALPHA_SPLIT, alpha, epsabs=3.0e-12, epsrel=3.0e-12,
                limit=500, points=points)[0]


C0 = first_integral_above(1.0)


def isotropic_radius_from_alpha(alpha: float) -> float:
    if alpha == 0.0:
        return 0.0
    if alpha >= ALPHA_SPLIT:
        return (radius_from_alpha(alpha)**(1.0 / alpha)
                * np.exp(first_integral_above(alpha) - C0))
    integral = quad(
        lambda value: -1.0 / (d_alpha_d_radius(
            value, radius_from_alpha(value)) * value * radius_from_alpha(value)),
        alpha, ALPHA_SPLIT, epsabs=3.0e-12, epsrel=3.0e-12, limit=500,
    )[0]
    return (radius_from_alpha(ALPHA_SPLIT)**(1.0 / ALPHA_SPLIT)
            * np.exp(integral - C0))


def build_table(count: int, minimum_r: float, maximum_r: float):
    log_r = np.linspace(np.log(minimum_r), np.log(maximum_r), count)
    r = np.exp(log_r)
    alpha_max = brentq(
        lambda alpha: isotropic_radius_from_alpha(alpha) - maximum_r,
        ALPHA_C, 1.0 - 1.0e-12, xtol=1.0e-14,
        rtol=4.0 * np.finfo(float).eps,
    )
    areal_max = radius_from_alpha(alpha_max)

    def ode(_log_r, state):
        areal = state[0]
        return [alpha_from_radius(areal) * areal]

    solution = solve_ivp(
        ode, (log_r[-1], log_r[0]), [areal_max],
        t_eval=log_r[::-1], method="DOP853", rtol=2.0e-13, atol=2.0e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    areal = solution.y[0, ::-1]
    alpha = np.array([alpha_from_radius(value) for value in areal])

    alpha_R, alpha_RR = alpha_radius_derivatives(alpha, areal)
    areal_d1 = alpha * areal / r
    alpha_d1 = alpha_R * areal_d1
    areal_d2 = ((alpha_d1 * areal + alpha * areal_d1) / r
                - alpha * areal / r**2)
    alpha_d2 = alpha_RR * areal_d1**2 + alpha_R * areal_d2

    shift_q = np.sqrt(C_SQUARED) * np.exp(alpha / N) / areal**3
    log_q_d1 = alpha_d1 / N - 3.0 * areal_d1 / areal
    log_q_d2 = (alpha_d2 / N
                - 3.0 * (areal_d2 / areal - (areal_d1 / areal)**2))
    shift_q_d1 = shift_q * log_q_d1
    shift_q_d2 = shift_q * (log_q_d1**2 + log_q_d2)

    residual = float(np.max(np.abs(implicit(alpha, areal))
                            / np.maximum(1.0, areal**4)))
    normalization = isotropic_radius_from_alpha(alpha_max)
    if residual > 2.0e-11:
        raise AssertionError(f"implicit trumpet residual too large: {residual}")
    if abs(normalization - maximum_r) > 2.0e-10 * maximum_r:
        raise AssertionError(f"isotropic normalization mismatch: {normalization}")
    def y_jet(value, radial_first, radial_second):
        return value, r * radial_first, r * radial_first + r**2 * radial_second

    return log_r, np.array([
        *y_jet(alpha, alpha_d1, alpha_d2),
        *y_jet(areal, areal_d1, areal_d2),
        *y_jet(shift_q, shift_q_d1, shift_q_d2),
    ]), residual, alpha_max, areal_max


def format_array(name: str, values: np.ndarray) -> str:
    lines = [f"inline constexpr Real {name}[kTrumpetTableSize] = {{"]
    for start in range(0, len(values), 4):
        chunk = ", ".join(f"{value:.17e}" for value in values[start:start + 4])
        lines.append(f"  {chunk},")
    lines.append("};")
    return "\n".join(lines)


def write_header(path: Path, count: int, minimum_r: float, maximum_r: float):
    log_r, data, residual, alpha_max, areal_max = build_table(
        count, minimum_r, maximum_r)
    names = ("kTrumpetAlpha", "kTrumpetAlphaDy", "kTrumpetAlphaDyy",
             "kTrumpetArealRadius", "kTrumpetArealRadiusDy",
             "kTrumpetArealRadiusDyy", "kTrumpetShiftQ",
             "kTrumpetShiftQDy", "kTrumpetShiftQDyy")
    arrays = "\n\n".join(format_array(name, values)
                           for name, values in zip(names, data))
    content = f"""// Generated by tst/test_suite/ref_gh/generate_trumpet_table.py.
// Do not edit by hand. Unit mass, stationary n=2 1+log trumpet.
#ifndef REF_GH_TRUMPET_TABLE_GENERATED_HPP_
#define REF_GH_TRUMPET_TABLE_GENERATED_HPP_

#include \"athena.hpp\"

namespace ref_gh {{
inline constexpr int kTrumpetTableSize = {count};
inline constexpr Real kTrumpetLogRMin = {log_r[0]:.17e};
inline constexpr Real kTrumpetLogRMax = {log_r[-1]:.17e};
inline constexpr Real kTrumpetLogRSpacing = {(log_r[1] - log_r[0]):.17e};
inline constexpr Real kTrumpetCriticalLapse = {ALPHA_C:.17e};
inline constexpr Real kTrumpetCriticalArealRadius = {RADIUS_C:.17e};
inline constexpr Real kTrumpetCSquared = {C_SQUARED:.17e};
inline constexpr Real kTrumpetR0 = {radius_from_alpha(0.0):.17e};
inline constexpr Real kTrumpetTableImplicitResidual = {residual:.17e};
inline constexpr Real kTrumpetAlphaAtRMax = {alpha_max:.17e};
inline constexpr Real kTrumpetArealRadiusAtRMax = {areal_max:.17e};

{arrays}

}}  // namespace ref_gh
#endif  // REF_GH_TRUMPET_TABLE_GENERATED_HPP_
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1025)
    parser.add_argument("--minimum-r", type=float, default=1.0e-4)
    parser.add_argument("--maximum-r", type=float, default=128.0)
    args = parser.parse_args()
    write_header(args.output, args.count, args.minimum_r, args.maximum_r)


if __name__ == "__main__":
    main()
