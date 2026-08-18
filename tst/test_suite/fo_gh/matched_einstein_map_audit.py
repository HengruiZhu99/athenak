#!/usr/bin/env python3
"""Finite-radius 58D chart oracle for the matched FO-GH state.

This is an algebra/map audit, not a production RHS.  It constructs an explicit
58-component chart satisfying det(gtilde)=1, tr(Atilde)=0, and
tr(Q_k)=0; maps it to {psi, Pi, Phi, H, Z}; and maps back independently.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np


SYM3_FREE = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2))
SYM4 = tuple((a, b) for a in range(4) for b in range(a, 4))


@dataclass
class RegularChart:
    g: np.ndarray
    chi: float
    A: float
    beta: np.ndarray
    At: np.ndarray
    K: float
    Lambda: np.ndarray
    pi: float
    Q: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    B: np.ndarray
    h: np.ndarray
    z: np.ndarray


@dataclass
class ParentState:
    psi: np.ndarray
    Pi: np.ndarray
    Phi: np.ndarray
    H: np.ndarray
    Z: np.ndarray


def conformal_metric(free: Sequence[float]) -> np.ndarray:
    g = np.zeros((3, 3), dtype=float)
    for value, (i, j) in zip(free, SYM3_FREE):
        g[i, j] = g[j, i] = value
    denominator = g[0, 0] * g[1, 1] - g[0, 1] ** 2
    numerator = (1.0 + g[0, 0] * g[1, 2] ** 2
                 - 2.0 * g[0, 1] * g[0, 2] * g[1, 2]
                 + g[0, 2] ** 2 * g[1, 1])
    g[2, 2] = numerator / denominator
    return g


def free_metric(g: np.ndarray) -> np.ndarray:
    return np.array([g[i, j] for i, j in SYM3_FREE])


def tracefree_tensor(free: Sequence[float], ginv: np.ndarray) -> np.ndarray:
    tensor = np.zeros((3, 3), dtype=float)
    for value, (i, j) in zip(free, SYM3_FREE):
        tensor[i, j] = tensor[j, i] = value
    partial_trace = 0.0
    for i in range(3):
        for j in range(3):
            if i == 2 and j == 2:
                continue
            partial_trace += ginv[i, j] * tensor[i, j]
    tensor[2, 2] = -partial_trace / ginv[2, 2]
    return tensor


def free_tensor(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[i, j] for i, j in SYM3_FREE])


def unpack_regular(vector: Sequence[float]) -> RegularChart:
    v = np.asarray(vector, dtype=float)
    if v.shape != (58,):
        raise ValueError("regular chart must have 58 components")
    offset = 0

    def take(count: int) -> np.ndarray:
        nonlocal offset
        result = v[offset:offset + count]
        offset += count
        return result

    g = conformal_metric(take(5))
    ginv = np.linalg.inv(g)
    chi, A = take(2)
    beta = take(3).copy()
    At = tracefree_tensor(take(5), ginv)
    K = float(take(1)[0])
    Lambda = take(3).copy()
    pi = float(take(1)[0])
    Q = np.stack([tracefree_tensor(take(5), ginv) for _ in range(3)])
    X = take(3).copy()
    Y = take(3).copy()
    B = take(9).reshape(3, 3).copy()
    h = take(4).copy()
    z = take(4).copy()
    if offset != 58:
        raise AssertionError(offset)
    return RegularChart(g, float(chi), float(A), beta, At, K, Lambda, pi,
                        Q, X, Y, B, h, z)


def pack_regular(state: RegularChart) -> np.ndarray:
    return np.concatenate((
        free_metric(state.g), [state.chi, state.A], state.beta,
        free_tensor(state.At), [state.K], state.Lambda, [state.pi],
        *(free_tensor(state.Q[k]) for k in range(3)),
        state.X, state.Y, state.B.reshape(9), state.h, state.z,
    ))


def pack_symmetric4(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[a, b] for a, b in SYM4])


def unpack_symmetric4(vector: Sequence[float]) -> np.ndarray:
    result = np.zeros((4, 4), dtype=float)
    for value, (a, b) in zip(vector, SYM4):
        result[a, b] = result[b, a] = value
    return result


def pack_parent(state: ParentState) -> np.ndarray:
    return np.concatenate((
        pack_symmetric4(state.psi), pack_symmetric4(state.Pi),
        *(pack_symmetric4(state.Phi[k]) for k in range(3)),
        state.H, state.Z,
    ))


def unpack_parent(vector: Sequence[float]) -> ParentState:
    v = np.asarray(vector, dtype=float)
    if v.shape != (58,):
        raise ValueError("parent state must have 58 components")
    offset = 0

    def take(count: int) -> np.ndarray:
        nonlocal offset
        result = v[offset:offset + count]
        offset += count
        return result

    psi = unpack_symmetric4(take(10))
    Pi = unpack_symmetric4(take(10))
    Phi = np.stack([unpack_symmetric4(take(10)) for _ in range(3)])
    H = take(4).copy()
    Z = take(4).copy()
    if offset != 58:
        raise AssertionError(offset)
    return ParentState(psi, Pi, Phi, H, Z)


def regular_to_parent(state: RegularChart) -> ParentState:
    if state.A <= 0.0 or state.chi <= 0.0:
        raise ValueError("A and chi must be positive")
    alpha = math.sqrt(state.A)
    gamma = state.g / state.chi
    gamma_inv = np.linalg.inv(gamma)
    Kij = (state.At + state.g * state.K / 3.0) / state.chi

    dgamma = np.empty((3, 3, 3))
    for k in range(3):
        dgamma[k] = state.Q[k] / state.chi \
            - state.X[k] * state.g / state.chi ** 2

    d0gamma = -2.0 * alpha * Kij
    for i in range(3):
        for j in range(3):
            d0gamma[i, j] += gamma[i] @ state.B[j] + gamma[j] @ state.B[i]

    ginv = np.linalg.inv(state.g)
    d0beta = (ginv @ state.h[1:] + state.A * state.chi * state.Lambda
              + 0.5 * state.A * (ginv @ state.X)
              - 0.5 * state.chi * (ginv @ state.Y))
    d0alpha = state.A * state.pi \
        - alpha * state.h[0] / (state.A * state.chi)

    psi = np.zeros((4, 4))
    psi[1:, 1:] = gamma
    beta_lower = gamma @ state.beta
    psi[0, 1:] = psi[1:, 0] = beta_lower
    psi[0, 0] = -state.A + state.beta @ beta_lower

    d0g = np.zeros((4, 4))
    d0g[1:, 1:] = d0gamma
    d0g[0, 1:] = d0g[1:, 0] = d0gamma @ state.beta + gamma @ d0beta
    d0g[0, 0] = (-2.0 * alpha * d0alpha
                  + state.beta @ d0gamma @ state.beta
                  + 2.0 * state.beta @ gamma @ d0beta)
    Pi = -d0g / alpha

    Phi = np.zeros((3, 4, 4))
    for k in range(3):
        Phi[k, 1:, 1:] = dgamma[k]
        Phi[k, 0, 1:] = Phi[k, 1:, 0] = (
            dgamma[k] @ state.beta + gamma @ state.B[k])
        Phi[k, 0, 0] = (-state.Y[k]
                         + state.beta @ dgamma[k] @ state.beta
                         + 2.0 * state.beta @ gamma @ state.B[k])

    w = state.A * state.chi
    H = np.concatenate(([(state.h[0] + state.beta @ state.h[1:]) / w],
                        state.h[1:] / w))
    Z = np.concatenate(([(state.z[0] + state.beta @ state.z[1:]) / w],
                        state.z[1:] / w))
    return ParentState(psi, Pi, Phi, H, Z)


def parent_to_regular(parent: ParentState) -> RegularChart:
    gamma = parent.psi[1:, 1:].copy()
    gamma_inv = np.linalg.inv(gamma)
    beta = gamma_inv @ parent.psi[0, 1:]
    A = -parent.psi[0, 0] + beta @ gamma @ beta
    if A <= 0.0:
        raise ValueError("recovered A is not positive")
    alpha = math.sqrt(A)
    chi = float(np.linalg.det(gamma) ** (-1.0 / 3.0))
    g = chi * gamma
    ginv = np.linalg.inv(g)

    dgamma = parent.Phi[:, 1:, 1:].copy()
    X = np.empty(3)
    Q = np.empty((3, 3, 3))
    B = np.empty((3, 3))
    Y = np.empty(3)
    for k in range(3):
        X[k] = -(chi / 3.0) * np.sum(gamma_inv * dgamma[k])
        Q[k] = X[k] * gamma + chi * dgamma[k]
        B[k] = gamma_inv @ (parent.Phi[k, 0, 1:] - dgamma[k] @ beta)
        Y[k] = (-parent.Phi[k, 0, 0] + beta @ dgamma[k] @ beta
                + 2.0 * beta @ gamma @ B[k])

    d0g = -alpha * parent.Pi
    d0gamma = d0g[1:, 1:]
    Kij = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            shift_lie = gamma[i] @ B[j] + gamma[j] @ B[i]
            Kij[i, j] = -(d0gamma[i, j] - shift_lie) / (2.0 * alpha)
    K = float(np.sum(gamma_inv * Kij))
    At = chi * (Kij - gamma * K / 3.0)

    d0beta = gamma_inv @ (d0g[0, 1:] - d0gamma @ beta)
    d0alpha = (-d0g[0, 0] + beta @ d0gamma @ beta
               + 2.0 * beta @ gamma @ d0beta) / (2.0 * alpha)

    w = A * chi
    h = np.concatenate(([w * (parent.H[0] - beta @ parent.H[1:])],
                        w * parent.H[1:]))
    z = np.concatenate(([w * (parent.Z[0] - beta @ parent.Z[1:])],
                        w * parent.Z[1:]))
    pi = (d0alpha + alpha * h[0] / w) / A
    Lambda = (d0beta - ginv @ h[1:] - 0.5 * A * (ginv @ X)
              + 0.5 * chi * (ginv @ Y)) / w
    return RegularChart(g, chi, A, beta, At, K, Lambda, pi, Q, X, Y, B, h, z)


def random_regular(rng: np.random.Generator) -> np.ndarray:
    gfree = np.array([
        rng.uniform(0.8, 1.3), rng.uniform(-0.08, 0.08),
        rng.uniform(-0.08, 0.08), rng.uniform(0.8, 1.3),
        rng.uniform(-0.08, 0.08),
    ])
    return np.concatenate((
        gfree,
        rng.uniform(0.2, 2.0, 2),
        rng.uniform(-0.3, 0.3, 3),
        rng.uniform(-0.2, 0.2, 5),
        rng.uniform(-0.4, 0.4, 1),
        rng.uniform(-0.4, 0.4, 3),
        rng.uniform(-0.4, 0.4, 1),
        rng.uniform(-0.2, 0.2, 15),
        rng.uniform(-0.3, 0.3, 3),
        rng.uniform(-0.3, 0.3, 3),
        rng.uniform(-0.3, 0.3, 9),
        rng.uniform(-0.4, 0.4, 4),
        rng.uniform(-0.4, 0.4, 4),
    ))


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)) / max(1.0, np.max(np.abs(right))))


def verify_round_trip(samples: int, seed: int = 191733) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    worst_regular = 0.0
    worst_parent = 0.0
    worst_constraint = 0.0
    for _ in range(samples):
        regular_vector = random_regular(rng)
        regular = unpack_regular(regular_vector)
        parent = regular_to_parent(regular)
        recovered = parent_to_regular(parent)
        recovered_vector = pack_regular(recovered)
        worst_regular = max(worst_regular,
                            relative_error(recovered_vector, regular_vector))
        parent_vector = pack_parent(parent)
        parent_again = pack_parent(regular_to_parent(recovered))
        worst_parent = max(worst_parent,
                           relative_error(parent_again, parent_vector))
        ginv = np.linalg.inv(regular.g)
        residuals = [abs(np.linalg.det(regular.g) - 1.0),
                     abs(np.sum(ginv * regular.At))]
        residuals.extend(abs(np.sum(ginv * regular.Q[k])) for k in range(3))
        worst_constraint = max(worst_constraint, *residuals)
    if worst_regular > 2.0e-12 or worst_parent > 2.0e-12:
        raise AssertionError(
            f"58D round trip failed: regular={worst_regular}, parent={worst_parent}")
    if worst_constraint > 2.0e-13:
        raise AssertionError(f"chart constraint residual: {worst_constraint}")
    return worst_regular, worst_parent, worst_constraint


def numerical_jacobian(function, point: np.ndarray) -> np.ndarray:
    baseline = np.asarray(function(point))
    jacobian = np.empty((baseline.size, point.size))
    for column in range(point.size):
        scale = max(1.0, abs(point[column]))
        step = 2.0e-6 * scale
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (function(plus) - function(minus)) / (2.0 * step)
    return jacobian


def verify_tangent_round_trip(samples: int, seed: int = 551239) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    worst_identity = 0.0
    worst_condition = 0.0
    for _ in range(samples):
        regular = random_regular(rng)
        parent = pack_parent(regular_to_parent(unpack_regular(regular)))
        J_forward = numerical_jacobian(
            lambda value: pack_parent(regular_to_parent(unpack_regular(value))),
            regular)
        J_inverse = numerical_jacobian(
            lambda value: pack_regular(parent_to_regular(unpack_parent(value))),
            parent)
        worst_identity = max(
            worst_identity,
            relative_error(J_inverse @ J_forward, np.eye(58)),
            relative_error(J_forward @ J_inverse, np.eye(58)),
        )
        worst_condition = max(worst_condition, float(np.linalg.cond(J_forward)))
    if worst_identity > 2.0e-6:
        raise AssertionError(f"tangent round trip failed: {worst_identity}")
    return worst_identity, worst_condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--tangent-samples", type=int, default=8)
    args = parser.parse_args()
    round_regular, round_parent, constraints = verify_round_trip(args.samples)
    tangent, condition = verify_tangent_round_trip(args.tangent_samples)
    print(f"regular_round_trip_max_relative_error={round_regular:.17e}")
    print(f"parent_round_trip_max_relative_error={round_parent:.17e}")
    print(f"chart_constraint_max_absolute_residual={constraints:.17e}")
    print(f"tangent_round_trip_max_relative_error={tangent:.17e}")
    print(f"random_finite_radius_jacobian_max_condition={condition:.17e}")
    print("MATCHED_EINSTEIN_MAP_ORACLE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
