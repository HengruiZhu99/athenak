#!/usr/bin/env python3
"""Audit the exact weighted pullback of the improved GH gauge driver.

The standard driver evolves spacetime covectors H_a and
Z_a = theta_a + eta H_a.  The proposed regular variables are

  h_perp = alpha H_perp = H_0 - beta^i H_i,
  h^i    = A H^i = A chi gtilde^{ij} H_j,

and the same weighted projections for z.  This module derives the directional
derivative of that linear weight/projection map as a 4x4 matrix, checks the
component formulas against independent dense linear algebra, and audits their
stationary-trumpet powers.  It intentionally has no AthenaK or SymPy dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, getcontext
import math
import random
from typing import List, Sequence, Tuple


Vector = List[float]
Matrix = List[List[float]]


def eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matvec(matrix: Matrix, vector: Sequence[float]) -> Vector:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    return [a + b for a, b in zip(left, right)]


def subtract(left: Sequence[float], right: Sequence[float]) -> Vector:
    return [a - b for a, b in zip(left, right)]


def scale(factor: float, vector: Sequence[float]) -> Vector:
    return [factor * value for value in vector]


def invert3(matrix: Matrix) -> Matrix:
    a, b, c = matrix[0]
    _, d, e = matrix[1]
    _, _, f = matrix[2]
    # Inputs are symmetric; use their lower triangle rather than assuming it
    # has already been copied by the caller.
    b = matrix[1][0] = matrix[0][1]
    c = matrix[2][0] = matrix[0][2]
    e = matrix[2][1] = matrix[1][2]
    det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    if det <= 0.0:
        raise ValueError("metric must be positive definite")
    return [
        [(d * f - e * e) / det, (c * e - b * f) / det,
         (b * e - c * d) / det],
        [(c * e - b * f) / det, (a * f - c * c) / det,
         (b * c - a * e) / det],
        [(b * e - c * d) / det, (b * c - a * e) / det,
         (a * d - b * b) / det],
    ]


def weight_matrix(a_sq: float, chi: float, gtilde: Matrix,
                  beta: Sequence[float]) -> Matrix:
    inverse = invert3([row[:] for row in gtilde])
    result = [[0.0] * 4 for _ in range(4)]
    result[0][0] = 1.0
    for i in range(3):
        result[0][i + 1] = -beta[i]
        for j in range(3):
            result[i + 1][j + 1] = a_sq * chi * inverse[i][j]
    return result


def weight_directional_derivative(
        a_sq: float, chi: float, gtilde: Matrix,
        da_sq: float, dchi: float, dgtilde: Matrix,
        dbeta: Sequence[float]) -> Matrix:
    inverse = invert3([row[:] for row in gtilde])
    inverse_dg_inverse = matmul(matmul(inverse, dgtilde), inverse)
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        result[0][i + 1] = -dbeta[i]
        for j in range(3):
            result[i + 1][j + 1] = (
                (da_sq * chi + a_sq * dchi) * inverse[i][j]
                - a_sq * chi * inverse_dg_inverse[i][j])
    return result


def covector_from_regular(weight: Matrix, regular: Sequence[float]) -> Vector:
    # The map is square and block triangular.  Inverting it explicitly keeps
    # this oracle independent of the derived component RHS below.
    spatial_block = [row[1:] for row in weight[1:]]
    spatial_inverse = invert3(spatial_block)
    covector_spatial = matvec(spatial_inverse, regular[1:])
    return [regular[0] - sum(weight[0][i + 1] * covector_spatial[i]
                             for i in range(3))] + covector_spatial


@dataclass
class DriverState:
    a_sq: float
    chi: float
    gtilde: Matrix
    beta: Vector
    h: Vector
    z: Vector
    f: Vector


@dataclass
class MetricDerivative:
    a_sq: float
    chi: float
    gtilde: Matrix
    beta: Vector


def derived_h_rhs(state: DriverState, d0: MetricDerivative,
                  mu: float, eta: float) -> Vector:
    inverse = invert3([row[:] for row in state.gtilde])
    h_covector = covector_from_regular(
        weight_matrix(state.a_sq, state.chi, state.gtilde, state.beta), state.h)
    spatial_mix = matmul(inverse, d0.gtilde)
    log_weight = d0.a_sq / state.a_sq + d0.chi / state.chi
    rhs = [state.z[0] - (mu + eta) * state.h[0] + mu * state.f[0]
           - sum(d0.beta[i] * h_covector[i + 1] for i in range(3))]
    for i in range(3):
        basis_term = log_weight * state.h[i + 1]
        basis_term -= sum(spatial_mix[i][j] * state.h[j + 1]
                          for j in range(3))
        rhs.append(state.z[i + 1] - (mu + eta) * state.h[i + 1]
                   + mu * state.f[i + 1] + basis_term)
    return rhs


def derived_z_rhs(state: DriverState, dt: MetricDerivative,
                  mu: float, eta: float) -> Vector:
    inverse = invert3([row[:] for row in state.gtilde])
    z_covector = covector_from_regular(
        weight_matrix(state.a_sq, state.chi, state.gtilde, state.beta), state.z)
    spatial_mix = matmul(inverse, dt.gtilde)
    log_weight = dt.a_sq / state.a_sq + dt.chi / state.chi
    rhs = [-sum(dt.beta[i] * z_covector[i + 1] for i in range(3))
           - eta * mu * (state.h[0] - state.f[0])]
    for i in range(3):
        basis_term = log_weight * state.z[i + 1]
        basis_term -= sum(spatial_mix[i][j] * state.z[j + 1]
                          for j in range(3))
        rhs.append(basis_term - eta * mu *
                   (state.h[i + 1] - state.f[i + 1]))
    return rhs


def dense_oracle_h_rhs(state: DriverState, d0: MetricDerivative,
                       mu: float, eta: float) -> Vector:
    weight = weight_matrix(state.a_sq, state.chi, state.gtilde, state.beta)
    dweight = weight_directional_derivative(
        state.a_sq, state.chi, state.gtilde,
        d0.a_sq, d0.chi, d0.gtilde, d0.beta)
    h_covector = covector_from_regular(weight, state.h)
    z_covector = covector_from_regular(weight, state.z)
    f_covector = covector_from_regular(weight, state.f)
    standard_rhs = add(subtract(z_covector, scale(mu + eta, h_covector)),
                       scale(mu, f_covector))
    return add(matvec(dweight, h_covector), matvec(weight, standard_rhs))


def dense_oracle_z_rhs(state: DriverState, dt: MetricDerivative,
                       mu: float, eta: float) -> Vector:
    weight = weight_matrix(state.a_sq, state.chi, state.gtilde, state.beta)
    dweight = weight_directional_derivative(
        state.a_sq, state.chi, state.gtilde,
        dt.a_sq, dt.chi, dt.gtilde, dt.beta)
    z_covector = covector_from_regular(weight, state.z)
    return add(matvec(dweight, z_covector),
               scale(-eta * mu, subtract(state.h, state.f)))


def random_spd(rng: random.Random) -> Matrix:
    lower = [[1.0 + rng.random(), 0.0, 0.0],
             [rng.uniform(-0.2, 0.2), 1.0 + rng.random(), 0.0],
             [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
              1.0 + rng.random()]]
    return matmul(lower, transpose(lower))


def random_symmetric(rng: random.Random, scale_value: float = 0.2) -> Matrix:
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            result[i][j] = result[j][i] = rng.uniform(-scale_value, scale_value)
    return result


def verify_dense_oracle(samples: int = 128, seed: int = 20260818) -> float:
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(samples):
        state = DriverState(
            a_sq=rng.uniform(0.2, 2.0), chi=rng.uniform(0.2, 2.0),
            gtilde=random_spd(rng),
            beta=[rng.uniform(-0.4, 0.4) for _ in range(3)],
            h=[rng.uniform(-1.0, 1.0) for _ in range(4)],
            z=[rng.uniform(-1.0, 1.0) for _ in range(4)],
            f=[rng.uniform(-1.0, 1.0) for _ in range(4)])
        d0 = MetricDerivative(
            a_sq=rng.uniform(-0.3, 0.3), chi=rng.uniform(-0.3, 0.3),
            gtilde=random_symmetric(rng),
            beta=[rng.uniform(-0.3, 0.3) for _ in range(3)])
        dt = MetricDerivative(
            a_sq=rng.uniform(-0.3, 0.3), chi=rng.uniform(-0.3, 0.3),
            gtilde=random_symmetric(rng),
            beta=[rng.uniform(-0.3, 0.3) for _ in range(3)])
        mu = rng.uniform(0.1, 2.0)
        eta = rng.uniform(0.1, 2.0)
        pairs = ((derived_h_rhs(state, d0, mu, eta),
                  dense_oracle_h_rhs(state, d0, mu, eta)),
                 (derived_z_rhs(state, dt, mu, eta),
                  dense_oracle_z_rhs(state, dt, mu, eta)))
        for derived, oracle in pairs:
            scale_value = max(1.0, max(abs(value) for value in oracle))
            worst = max(worst, max(abs(a - b) for a, b in zip(derived, oracle))
                        / scale_value)
    if worst > 2.0e-14:
        raise AssertionError("weighted-driver dense oracle mismatch: {}".format(worst))
    return worst


def verify_regular_gauge_target(samples: int = 128, seed: int = 20260819) -> float:
    """Check the requested 1+log and integrated Gamma-driver identities."""
    rng = random.Random(seed)
    worst = 0.0
    nu = 0.75
    for _ in range(samples):
        a_sq = rng.uniform(0.1, 2.0)
        alpha = math.sqrt(a_sq)
        chi = rng.uniform(0.1, 2.0)
        gtilde_inverse = invert3(random_spd(rng))
        pi = rng.uniform(-1.0, 1.0)
        trace_k = rng.uniform(-1.0, 1.0)
        eta_beta = rng.uniform(0.1, 2.0)
        beta = [rng.uniform(-0.4, 0.4) for _ in range(3)]
        lambda_up = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        x_gradient = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        y_gradient = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        f_perp = alpha * pi + 2.0 * trace_k
        f_up = []
        for i in range(3):
            gradient = sum(gtilde_inverse[i][j] *
                           (-0.5 * a_sq * x_gradient[j]
                            + 0.5 * chi * y_gradient[j])
                           for j in range(3))
            f_up.append((nu - a_sq * chi) * lambda_up[i] + gradient
                        - eta_beta * beta[i])
        d0_a_sq = 2.0 * a_sq * (alpha * pi - f_perp)
        d0_beta = []
        for i in range(3):
            geometric = a_sq * chi * lambda_up[i]
            geometric += sum(gtilde_inverse[i][j] *
                             (0.5 * a_sq * x_gradient[j]
                              - 0.5 * chi * y_gradient[j])
                             for j in range(3))
            d0_beta.append(f_up[i] + geometric)
        scale_value = max(1.0, abs(4.0 * a_sq * trace_k),
                          *(abs(nu * lambda_up[i] - eta_beta * beta[i])
                            for i in range(3)))
        errors = [abs(d0_a_sq + 4.0 * a_sq * trace_k)]
        errors.extend(abs(d0_beta[i] -
                          (nu * lambda_up[i] - eta_beta * beta[i]))
                      for i in range(3))
        worst = max(worst, max(errors) / scale_value)
    if worst > 3.0e-15:
        raise AssertionError("regular gauge target mismatch: {}".format(worst))
    return worst


def trumpet_power_audit(p: float = 1.091) -> Tuple[float, float]:
    # Generic regular Cartesian vector fields beta, h and D0 beta are O(r).
    denominator_power = 2.0 * p + 2.0  # A chi
    numerator_power = 1.0 + 1.0        # (D0 beta)^i h^j
    bad_power = numerator_power - denominator_power
    required_z_perp_power = bad_power
    return bad_power, required_z_perp_power


def conditioning_sequence(p: str = "1.091", maximum_n: int = 80):
    getcontext().prec = 100
    decimal_p = Decimal(p)
    coefficient = Decimal("0.15")
    rows = []
    for n in range(1, maximum_n + 1):
        radius = Decimal(2) ** Decimal(-n)
        high_precision = coefficient * (-Decimal(2) * decimal_p * radius.ln()).exp()
        radius_float = math.ldexp(1.0, -n)
        double_value = 0.15 * radius_float ** (-2.0 * float(p))
        relative_error = abs(Decimal.from_float(double_value) - high_precision) / high_precision
        rows.append((n, radius, double_value, high_precision, relative_error))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--maximum-n", type=int, default=80)
    args = parser.parse_args()
    mismatch = verify_dense_oracle(args.samples)
    target_mismatch = verify_regular_gauge_target(args.samples)
    bad_power, z_power = trumpet_power_audit()
    sequence = conditioning_sequence(maximum_n=args.maximum_n)
    print("weighted_driver_dense_oracle=PASS")
    print("maximum_relative_mismatch={:.17e}".format(mismatch))
    print("regular_gauge_target_oracle=PASS")
    print("gauge_target_maximum_relative_mismatch={:.17e}".format(target_mismatch))
    print("normal_h_projection_term_power={:.6f}".format(bad_power))
    print("required_z_perp_power={:.6f}".format(z_power))
    for n, radius, double_value, high_precision, relative_error in sequence:
        if n in (1, 8, 16, 32, 48, 64, args.maximum_n):
            print("n={} r={} double={:.17e} reference={} relerr={}".format(
                n, radius, double_value, high_precision, relative_error))
    if bad_power < 0.0 or z_power < 0.0:
        print("FORMULATION NOT ESTABLISHED: exact weighted driver requires a "
              "divergent normal projection or divergent z_perp")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
