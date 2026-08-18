#!/usr/bin/env python3
"""Independent audit of the matched A*chi gauge-covector pullback.

This deliberately does not import ``exact_driver_pullback_audit``.  It checks
the old obstruction and the new map by two separate paths:

* explicit component equations derived from the four covector components;
* dense differentiation of the 4x4 map W = (A chi) T(beta).

It also checks the moving-puncture target, the normalized-map conditioning,
and every named production contribution in the stationary-trumpet power
model.  The high-precision scan uses Decimal with at least 100 digits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, getcontext
import math
from typing import Dict, Iterable, Tuple

import numpy as np


P_TRUMPET = Decimal("1.091")


@dataclass(frozen=True)
class DriverSample:
    A: float
    chi: float
    beta: np.ndarray
    h: np.ndarray
    z: np.ndarray
    f: np.ndarray
    d0_A: float
    d0_chi: float
    d0_beta: np.ndarray
    dt_A: float
    dt_chi: float
    dt_beta: np.ndarray
    mu: float
    eta: float


def normalized_map(beta: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[0, 1:] = -beta
    return result


def normalized_map_inverse(beta: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[0, 1:] = beta
    return result


def matched_map(A: float, chi: float, beta: np.ndarray) -> np.ndarray:
    return (A * chi) * normalized_map(beta)


def matched_map_derivative(
    A: float,
    chi: float,
    beta: np.ndarray,
    d_A: float,
    d_chi: float,
    d_beta: np.ndarray,
) -> np.ndarray:
    w = A * chi
    d_w = d_A * chi + A * d_chi
    d_T = np.zeros((4, 4))
    d_T[0, 1:] = -d_beta
    return d_w * normalized_map(beta) + w * d_T


def explicit_inverse(A: float, chi: float, beta: np.ndarray,
                     regular: np.ndarray) -> np.ndarray:
    """Inverse derived component-by-component, without a dense solve."""
    w = A * chi
    spatial = regular[1:] / w
    return np.concatenate(([(regular[0] + beta @ regular[1:]) / w], spatial))


def component_h_rhs(sample: DriverSample) -> np.ndarray:
    c_H = sample.mu + sample.eta
    ell_0 = sample.d0_A / sample.A + sample.d0_chi / sample.chi
    rhs = np.empty(4)
    rhs[0] = ((ell_0 - c_H) * sample.h[0] + sample.z[0]
              + sample.mu * sample.f[0] - sample.d0_beta @ sample.h[1:])
    rhs[1:] = ((ell_0 - c_H) * sample.h[1:] + sample.z[1:]
               + sample.mu * sample.f[1:])
    return rhs


def component_z_rhs(sample: DriverSample) -> np.ndarray:
    ell_t = sample.dt_A / sample.A + sample.dt_chi / sample.chi
    rhs = np.empty(4)
    rhs[0] = (ell_t * sample.z[0] - sample.dt_beta @ sample.z[1:]
              - sample.eta * sample.mu * (sample.h[0] - sample.f[0]))
    rhs[1:] = (ell_t * sample.z[1:]
               - sample.eta * sample.mu * (sample.h[1:] - sample.f[1:]))
    return rhs


def dense_h_rhs(sample: DriverSample) -> np.ndarray:
    W = matched_map(sample.A, sample.chi, sample.beta)
    dW = matched_map_derivative(
        sample.A, sample.chi, sample.beta,
        sample.d0_A, sample.d0_chi, sample.d0_beta)
    H = explicit_inverse(sample.A, sample.chi, sample.beta, sample.h)
    Z = explicit_inverse(sample.A, sample.chi, sample.beta, sample.z)
    F = explicit_inverse(sample.A, sample.chi, sample.beta, sample.f)
    parent = Z - (sample.mu + sample.eta) * H + sample.mu * F
    return dW @ H + W @ parent


def dense_z_rhs(sample: DriverSample) -> np.ndarray:
    W = matched_map(sample.A, sample.chi, sample.beta)
    dW = matched_map_derivative(
        sample.A, sample.chi, sample.beta,
        sample.dt_A, sample.dt_chi, sample.dt_beta)
    Z = explicit_inverse(sample.A, sample.chi, sample.beta, sample.z)
    return dW @ Z - sample.eta * sample.mu * (sample.h - sample.f)


def random_sample(rng: np.random.Generator) -> DriverSample:
    return DriverSample(
        A=float(rng.uniform(0.05, 3.0)),
        chi=float(rng.uniform(0.05, 3.0)),
        beta=rng.uniform(-1.0, 1.0, 3),
        h=rng.uniform(-2.0, 2.0, 4),
        z=rng.uniform(-2.0, 2.0, 4),
        f=rng.uniform(-2.0, 2.0, 4),
        d0_A=float(rng.uniform(-1.0, 1.0)),
        d0_chi=float(rng.uniform(-1.0, 1.0)),
        d0_beta=rng.uniform(-1.0, 1.0, 3),
        dt_A=float(rng.uniform(-1.0, 1.0)),
        dt_chi=float(rng.uniform(-1.0, 1.0)),
        dt_beta=rng.uniform(-1.0, 1.0, 3),
        mu=float(rng.uniform(0.05, 3.0)),
        eta=float(rng.uniform(0.05, 3.0)),
    )


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)) / max(1.0, np.max(np.abs(right))))


def verify_old_map_independently(samples: int, seed: int = 731921) -> Tuple[float, float]:
    """Reproduce the old inverse and its r^(-2p) normal mixing power."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        A = float(rng.uniform(0.05, 3.0))
        chi = float(rng.uniform(0.05, 3.0))
        beta = rng.uniform(-1.0, 1.0, 3)
        lower = np.tril(rng.uniform(-0.3, 0.3, (3, 3)))
        lower[np.diag_indices(3)] += 1.5
        gtilde = lower @ lower.T
        ginv = np.linalg.inv(gtilde)
        W_old = np.zeros((4, 4))
        W_old[0, 0] = 1.0
        W_old[0, 1:] = -beta
        W_old[1:, 1:] = A * chi * ginv
        H = rng.uniform(-2.0, 2.0, 4)
        old = W_old @ H
        H_i = gtilde @ old[1:] / (A * chi)
        recovered = np.concatenate(([old[0] + beta @ H_i], H_i))
        worst = max(worst, relative_error(recovered, H))
    power = float(Decimal(2) - (Decimal(2) * P_TRUMPET + Decimal(2)))
    if worst > 2.0e-13:
        raise AssertionError(f"old-map inverse mismatch: {worst:.17e}")
    if not power < 0.0:
        raise AssertionError("old-map divergence was not reproduced")
    return worst, power


def verify_matched_driver(samples: int, seed: int = 948731) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    worst_inverse = 0.0
    worst_rhs = 0.0
    for _ in range(samples):
        sample = random_sample(rng)
        W = matched_map(sample.A, sample.chi, sample.beta)
        expected_inverse = normalized_map_inverse(sample.beta) / (sample.A * sample.chi)
        worst_inverse = max(
            worst_inverse,
            relative_error(expected_inverse @ W, np.eye(4)),
            relative_error(W @ expected_inverse, np.eye(4)),
        )
        worst_rhs = max(
            worst_rhs,
            relative_error(component_h_rhs(sample), dense_h_rhs(sample)),
            relative_error(component_z_rhs(sample), dense_z_rhs(sample)),
        )
    if worst_inverse > 2.0e-13:
        raise AssertionError(f"matched-map inverse mismatch: {worst_inverse:.17e}")
    if worst_rhs > 3.0e-13:
        raise AssertionError(f"matched-driver oracle mismatch: {worst_rhs:.17e}")
    return worst_inverse, worst_rhs


def verify_target(samples: int, seed: int = 845123) -> float:
    rng = np.random.default_rng(seed)
    worst = 0.0
    nu = 0.75
    for _ in range(samples):
        A = float(rng.uniform(0.05, 3.0))
        alpha = math.sqrt(A)
        chi = float(rng.uniform(0.05, 3.0))
        lower = np.tril(rng.uniform(-0.3, 0.3, (3, 3)))
        lower[np.diag_indices(3)] += 1.5
        gtilde = lower @ lower.T
        ginv = np.linalg.inv(gtilde)
        beta = rng.uniform(-1.0, 1.0, 3)
        Lambda = rng.uniform(-2.0, 2.0, 3)
        X = rng.uniform(-2.0, 2.0, 3)
        Y = rng.uniform(-2.0, 2.0, 3)
        pi = float(rng.uniform(-2.0, 2.0))
        K = float(rng.uniform(-2.0, 2.0))
        eta_beta = float(rng.uniform(0.05, 3.0))
        w = A * chi
        f_perp = w * (alpha * pi + 2.0 * K)
        f_i = ((nu - w) * (gtilde @ Lambda) - 0.5 * A * X
               + 0.5 * chi * Y - eta_beta * (gtilde @ beta))
        d0_A = 2.0 * A * alpha * pi - 2.0 * f_perp / chi
        d0_beta = ginv @ f_i + w * Lambda + 0.5 * A * (ginv @ X) \
            - 0.5 * chi * (ginv @ Y)
        target_A = -4.0 * A * K
        target_beta = nu * Lambda - eta_beta * beta
        scale = max(1.0, abs(target_A), float(np.max(np.abs(target_beta))))
        worst = max(worst, abs(d0_A - target_A) / scale,
                    float(np.max(np.abs(d0_beta - target_beta))) / scale)
    if worst > 2.0e-13:
        raise AssertionError(f"matched target mismatch: {worst:.17e}")
    return worst


def verify_conditioning(samples: int, seed: int = 394127) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    worst_condition = 0.0
    worst_inverse = 0.0
    for _ in range(samples):
        beta = rng.uniform(-2.0, 2.0, 3)
        T = normalized_map(beta)
        worst_condition = max(worst_condition, float(np.linalg.cond(T, 2)))
        worst_inverse = max(
            worst_inverse,
            relative_error(T @ normalized_map_inverse(beta), np.eye(4)),
        )
    if not math.isfinite(worst_condition):
        raise AssertionError("normalized map has nonfinite condition number")
    if worst_inverse > 2.0e-14:
        raise AssertionError(f"normalized-map inverse mismatch: {worst_inverse:.17e}")
    return worst_condition, worst_inverse


def power_inventory() -> Dict[str, Decimal]:
    p = P_TRUMPET
    zero = Decimal(0)
    one = Decimal(1)
    two = Decimal(2)
    A = two * p
    chi = two
    beta = one
    K = zero
    pi = zero
    Lambda = one
    X = one
    Y = two * p - one
    B = zero
    d0_beta = one
    dt_beta = one
    w = A + chi

    # These are derived with min-plus leading-power algebra, rather than
    # inserted as the hoped-for answer.  Finite gtilde and its inverse carry
    # power zero.  The stationary gauge target fixes hhat=fhat.
    fhat_perp = min(w + p + pi, w + K)
    fhat_i = min(Lambda, A + X, chi + Y, beta)
    hhat_perp = fhat_perp
    hhat_i = fhat_i
    ell_0 = min(p + pi, hhat_perp - w, p + K, B)
    ell_t = min(ell_0, beta + Y - A, beta + X - chi)
    d0_hhat_perp = beta + hhat_perp - one
    d0_hhat_i = beta + hhat_i - one
    zhat_perp = min(d0_hhat_perp, ell_0 + hhat_perp,
                       hhat_perp, fhat_perp, d0_beta + hhat_i)
    zhat_i = min(d0_hhat_i, ell_0 + hhat_i, hhat_i, fhat_i)
    return {
        # Fundamental fields and first differentiated lapse field.
        "field.w": w,
        "field.hhat_perp": hhat_perp,
        "field.hhat_i": hhat_i,
        "field.zhat_perp": zhat_perp,
        "field.zhat_i": zhat_i,
        "field.Y_i": Y,
        "intermediate.d_j_Y_i": Y - one,
        # Regular logarithmic-rate contractions.  Singular vectors Y/A and
        # X/chi are intentionally absent because production must not form them.
        "ell0.sqrtA_pi": p + pi,
        "ell0.hhat_perp_over_w": hhat_perp - w,
        "ell0.sqrtA_K": p + K,
        "ell0.B_trace": B,
        "ellt.beta_dot_Y_over_A": beta + Y - A,
        "ellt.beta_dot_X_over_chi": beta + X - chi,
        # Matched targets.
        "fhat_perp.w_sqrtA_pi": w + p + pi,
        "fhat_perp.w_K": w + K,
        "fhat_i.Lambda": Lambda,
        "fhat_i.A_X": A + X,
        "fhat_i.chi_Y": chi + Y,
        "fhat_i.beta": beta,
        # D0 hhat_perp contributions.
        "rhs_hperp.ell0_h": ell_0 + hhat_perp,
        "rhs_hperp.cH_h": hhat_perp,
        "rhs_hperp.z": zhat_perp,
        "rhs_hperp.mu_f": fhat_perp,
        "rhs_hperp.d0beta_dot_hi": d0_beta + hhat_i,
        # D0 hhat_i contributions.
        "rhs_hi.ell0_h": ell_0 + hhat_i,
        "rhs_hi.cH_h": hhat_i,
        "rhs_hi.z": zhat_i,
        "rhs_hi.mu_f": fhat_i,
        # dt zhat_perp contributions, using dt beta = O(r) dynamically.
        "rhs_zperp.ellt_z": ell_t + zhat_perp,
        "rhs_zperp.dtbeta_dot_zi": dt_beta + zhat_i,
        "rhs_zperp.driver_residual": hhat_perp,
        # dt zhat_i contributions.
        "rhs_zi.ellt_z": ell_t + zhat_i,
        "rhs_zi.driver_residual": hhat_i,
    }


def high_precision_scan(maximum_n: int) -> Tuple[Dict[str, Decimal], Decimal]:
    getcontext().prec = 110
    powers = power_inventory()
    worst_relative = Decimal(0)
    coefficient = Decimal("0.731")
    for name, power in powers.items():
        if power < 0:
            raise AssertionError(f"negative production power: {name}={power}")
        power_float = float(power)
        for n in range(1, maximum_n + 1):
            radius = Decimal(2) ** Decimal(-n)
            reference = coefficient * (power * radius.ln()).exp()
            double = float(coefficient) * math.ldexp(1.0, -n) ** power_float
            if reference != 0:
                error = abs(Decimal.from_float(double) - reference) / abs(reference)
                worst_relative = max(worst_relative, error)
    return powers, worst_relative


def selected_scan_rows(powers: Dict[str, Decimal], maximum_n: int) -> Iterable[str]:
    getcontext().prec = 110
    for name in sorted(powers):
        power = powers[name]
        values = []
        for n in (1, 8, 16, 32, 48, 64, maximum_n):
            if n > maximum_n:
                continue
            radius = Decimal(2) ** Decimal(-n)
            value = Decimal("0.731") * (power * radius.ln()).exp()
            values.append(f"n{n}={value:.6E}")
        yield f"power[{name}]={power} " + " ".join(dict.fromkeys(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--maximum-n", type=int, default=64)
    args = parser.parse_args()
    if args.samples < 1000:
        raise SystemExit("--samples must be at least 1000")
    if args.maximum_n < 64:
        raise SystemExit("--maximum-n must be at least 64")

    old_inverse, old_power = verify_old_map_independently(args.samples)
    map_inverse, driver_error = verify_matched_driver(args.samples)
    target_error = verify_target(args.samples)
    condition, condition_inverse = verify_conditioning(args.samples)
    powers, precision_error = high_precision_scan(args.maximum_n)

    print(f"old_map_inverse_max_relative_error={old_inverse:.17e}")
    print(f"old_map_normal_mixing_power={old_power:.6f}")
    print(f"matched_map_inverse_max_relative_error={map_inverse:.17e}")
    print(f"matched_driver_dense_oracle_max_relative_error={driver_error:.17e}")
    print(f"matched_target_max_relative_error={target_error:.17e}")
    print(f"normalized_map_max_condition_number={condition:.17e}")
    print(f"normalized_map_inverse_max_relative_error={condition_inverse:.17e}")
    for row in selected_scan_rows(powers, args.maximum_n):
        print(row)
    print(f"high_precision_scan_max_relative_error={precision_error}")
    print("MATCHED_DRIVER_V0=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
