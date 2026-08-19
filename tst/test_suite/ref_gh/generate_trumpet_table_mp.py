#!/usr/bin/env python3
"""Generate the production trumpet Hermite table at arbitrary precision.

The final values are rounded once to binary64 when written.  This removes the
ODE accumulation noise in the legacy SciPy generator before Hermite endpoint
data are formed.  It requires mpmath and deliberately does not read an existing
generated table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mpmath as mp
import numpy as np

try:
    from . import generate_trumpet_table as legacy
    from . import high_precision_trumpet_source_oracle as oracle
except ImportError:
    import generate_trumpet_table as legacy
    import high_precision_trumpet_source_oracle as oracle


def build_table(count: int, minimum_r: float, maximum_r: float, dps: int):
    mp.mp.dps = dps
    oracle.N = mp.mpf(2)
    oracle.ALPHA_C, oracle.RADIUS_C, oracle.C_SQUARED = oracle.constants()
    oracle.radius_from_alpha.zero_radius = oracle.bisect(
        lambda radius: oracle.implicit(mp.mpf(0), radius),
        mp.mpf(0), oracle.RADIUS_C, int(mp.mp.dps * 3.6) + 32)
    c0 = oracle.normalization_constant()
    split_alpha = mp.mpf("0.1")
    split_radius = oracle.radius_from_alpha(split_alpha)
    split_y = mp.log(split_radius) / split_alpha - c0

    def rhs(_y, alpha):
        areal = oracle.radius_from_alpha(alpha)
        areal_alpha = oracle.radius_alpha_derivatives(alpha, areal)[0]
        return alpha * areal / areal_alpha

    def rk4_step(y, alpha, step):
        k1 = rhs(y, alpha)
        k2 = rhs(y + step / 2, alpha + step * k1 / 2)
        k3 = rhs(y + step / 2, alpha + step * k2 / 2)
        k4 = rhs(y + step, alpha + step * k3)
        return alpha + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def rk4_substeps(y, alpha, step, count_steps):
        substep = step / count_steps
        for local in range(count_steps):
            alpha = rk4_step(y + local * substep, alpha, substep)
        return alpha

    def advance(y, alpha, step):
        one = rk4_substeps(y, alpha, step, 1)
        two = rk4_substeps(y, alpha, step, 2)
        four = rk4_substeps(y, alpha, step, 4)
        first = (16 * two - one) / 15
        second = (16 * four - two) / 15
        return (32 * second - first) / 31
    log_r_mp = [mp.log(mp.mpf(str(minimum_r)))
                + index * (mp.log(mp.mpf(str(maximum_r)))
                           - mp.log(mp.mpf(str(minimum_r)))) / (count - 1)
                for index in range(count)]
    alpha_values = [None] * count
    below = [index for index, y in enumerate(log_r_mp) if y < split_y]
    current_y = split_y
    current_alpha = split_alpha
    for index in reversed(below):
        target_y = log_r_mp[index]
        current_alpha = advance(current_y, current_alpha, target_y - current_y)
        current_y = target_y
        alpha_values[index] = current_alpha
    current_y = split_y
    current_alpha = split_alpha
    for index, target_y in enumerate(log_r_mp):
        if target_y >= split_y:
            current_alpha = advance(current_y, current_alpha, target_y - current_y)
            current_y = target_y
            alpha_values[index] = current_alpha

    rows = [[] for _ in range(9)]
    residual = mp.mpf(0)
    areal_values = []
    for y, alpha in zip(log_r_mp, alpha_values):
        radius = mp.exp(y)
        areal = oracle.radius_from_alpha(alpha)
        areal_alpha, areal_alpha2 = oracle.radius_alpha_derivatives(alpha, areal)
        alpha_areal = 1 / areal_alpha
        alpha_areal2 = -areal_alpha2 / areal_alpha**3
        areal_y = alpha * areal
        alpha_y = alpha_areal * areal_y
        areal_yy = alpha_y * areal + alpha * areal_y
        alpha_yy = alpha_areal2 * areal_y**2 + alpha_areal * areal_yy
        q = mp.sqrt(oracle.C_SQUARED) * mp.exp(alpha / oracle.N) / areal**3
        log_q_y = alpha_y / oracle.N - 3 * areal_y / areal
        log_q_yy = (alpha_yy / oracle.N
                    - 3 * (areal_yy / areal - (areal_y / areal)**2))
        q_y = q * log_q_y
        q_yy = q * (log_q_y**2 + log_q_yy)
        values = (alpha, alpha_y, alpha_yy, areal, areal_y, areal_yy,
                  q, q_y, q_yy)
        for row, value in zip(rows, values):
            row.append(float(value))
        residual = max(residual, abs(oracle.implicit(alpha, areal))
                        / max(mp.mpf(1), areal**4))
        areal_values.append(areal)
    return (np.array([float(value) for value in log_r_mp]), np.array(rows),
            float(residual), float(alpha_values[-1]), float(areal_values[-1]))


def write_header(path: Path, count: int, minimum_r: float,
                 maximum_r: float, dps: int):
    log_r, data, residual, alpha_max, areal_max = build_table(
        count, minimum_r, maximum_r, dps)
    names = ("kTrumpetAlpha", "kTrumpetAlphaDy", "kTrumpetAlphaDyy",
             "kTrumpetArealRadius", "kTrumpetArealRadiusDy",
             "kTrumpetArealRadiusDyy", "kTrumpetShiftQ",
             "kTrumpetShiftQDy", "kTrumpetShiftQDyy")
    arrays = "\n\n".join(legacy.format_array(name, values)
                           for name, values in zip(names, data))
    content = f"""// Generated by tst/test_suite/ref_gh/generate_trumpet_table_mp.py.
// Do not edit by hand. Unit mass, stationary n=2 1+log trumpet.
#ifndef REF_GH_TRUMPET_TABLE_GENERATED_HPP_
#define REF_GH_TRUMPET_TABLE_GENERATED_HPP_

#include \"athena.hpp\"

namespace ref_gh {{
inline constexpr int kTrumpetTableSize = {count};
inline constexpr Real kTrumpetLogRMin = {log_r[0]:.17e};
inline constexpr Real kTrumpetLogRMax = {log_r[-1]:.17e};
inline constexpr Real kTrumpetLogRSpacing = {(log_r[1] - log_r[0]):.17e};
inline constexpr Real kTrumpetCriticalLapse = {float(oracle.ALPHA_C):.17e};
inline constexpr Real kTrumpetCriticalArealRadius = {float(oracle.RADIUS_C):.17e};
inline constexpr Real kTrumpetCSquared = {float(oracle.C_SQUARED):.17e};
inline constexpr Real kTrumpetR0 = {float(oracle.radius_from_alpha.zero_radius):.17e};
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
    parser.add_argument("--count", type=int, default=4097)
    parser.add_argument("--minimum-r", type=float, default=1.0e-4)
    parser.add_argument("--maximum-r", type=float, default=128.0)
    parser.add_argument("--dps", type=int, default=80)
    args = parser.parse_args()
    if args.dps < 60:
        raise ValueError("production table generation requires at least 60 dps")
    write_header(args.output, args.count, args.minimum_r, args.maximum_r, args.dps)


if __name__ == "__main__":
    main()
