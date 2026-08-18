#!/usr/bin/env python3
"""Finite-radius principal/symmetrizer audit for the matched 58D chart."""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Tuple

import numpy as np

from fo_gh.matched_einstein_map_audit import (
    SYM4,
    pack_parent,
    regular_to_parent,
    unpack_parent,
    unpack_regular,
)


PAIRS = list(SYM4)


def parent_geometry(parent_vector: np.ndarray):
    state = unpack_parent(parent_vector)
    gamma = state.psi[1:, 1:]
    gamma_inverse = np.linalg.inv(gamma)
    beta = gamma_inverse @ state.psi[0, 1:]
    A = -state.psi[0, 0] + beta @ gamma @ beta
    return state, gamma, gamma_inverse, beta, math.sqrt(A)


def parent_principal_matrices(parent_vector: np.ndarray, gamma1: float = -1.0,
                              gamma2: float = 1.0):
    _, _, gamma_inverse, beta, alpha = parent_geometry(parent_vector)
    matrices = []
    for k in range(3):
        matrix = np.zeros((58, 58))
        for q, (a, b) in enumerate(PAIRS):
            matrix[q, q] = -(1.0 + gamma1) * beta[k]
            matrix[10 + q, 10 + q] = -beta[k]
            matrix[10 + q, q] = -gamma1 * gamma2 * beta[k]
            for i in range(3):
                matrix[10 + q, 20 + 10 * i + q] += alpha * gamma_inverse[k, i]

            # Substitute the driver equation for time-index derivatives of H:
            # partial_0 H = beta^k partial_k H + lower order.
            derivative_a = beta[k] if a == 0 else float(a - 1 == k)
            derivative_b = beta[k] if b == 0 else float(b - 1 == k)
            for c in range(4):
                matrix[10 + q, 50 + c] += alpha * (
                    derivative_a * float(b == c)
                    + derivative_b * float(a == c))

            for i in range(3):
                row = 20 + 10 * i + q
                matrix[row, row] = -beta[k]
                if i == k:
                    matrix[row, 10 + q] += alpha
                    matrix[row, q] -= alpha * gamma2
        for a in range(4):
            matrix[50 + a, 50 + a] = -beta[k]
            # Z has zero coordinate characteristic speed.
        matrices.append(matrix)
    return matrices


def unpack_perturbation(vector: np.ndarray):
    psi = np.zeros((4, 4))
    Pi = np.zeros((4, 4))
    Phi = np.zeros((3, 4, 4))
    for value, (a, b) in zip(vector[:10], PAIRS):
        psi[a, b] = psi[b, a] = value
    for value, (a, b) in zip(vector[10:20], PAIRS):
        Pi[a, b] = Pi[b, a] = value
    for i in range(3):
        for value, (a, b) in zip(vector[20 + 10 * i:30 + 10 * i], PAIRS):
            Phi[i, a, b] = Phi[i, b, a] = value
    return psi, Pi, Phi, vector[50:54], vector[54:58]


def parent_energy(vector: np.ndarray, parent_vector: np.ndarray,
                  gamma2: float = 1.0) -> float:
    state, _, gamma_inverse, _, _ = parent_geometry(parent_vector)
    dpsi, dPi, dPhi, dH, dZ = unpack_perturbation(vector)
    # Eq. (B16), choosing m_ab=delta_ab and all nonzero Lambda weights as 1.
    energy = float(np.sum(dpsi * dpsi) + dH @ dH + dZ @ dZ
                   + np.sum((dPi - gamma2 * dpsi) ** 2))
    projection = state.psi[1:, :]
    combined = np.empty_like(dPhi)
    for i in range(3):
        combined[i] = dPhi[i] + 2.0 * np.outer(projection[i], dH)
    energy += float(np.einsum("ij,iac,jac", gamma_inverse, combined, combined))
    return energy


def parent_symmetrizer(parent_vector: np.ndarray, gamma2: float = 1.0) -> np.ndarray:
    basis = np.eye(58)
    diagonal = np.array([parent_energy(row, parent_vector, gamma2) for row in basis])
    matrix = np.diag(diagonal)
    for i in range(58):
        for j in range(i + 1, 58):
            value = (parent_energy(basis[i] + basis[j], parent_vector, gamma2)
                     - diagonal[i] - diagonal[j]) / 2.0
            matrix[i, j] = matrix[j, i] = value
    return matrix


def forward_vector(regular_vector: np.ndarray) -> np.ndarray:
    return pack_parent(regular_to_parent(unpack_regular(regular_vector)))


def forward_jacobian(point: np.ndarray, relative_step: float = 2.0e-6) -> np.ndarray:
    baseline = forward_vector(point)
    jacobian = np.empty((58, 58))
    for column in range(58):
        # A and chi need a relative positive perturbation near a puncture.
        if column in (5, 6):
            step = relative_step * abs(point[column])
        else:
            step = relative_step * max(abs(point[column]), 1.0e-8)
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (forward_vector(plus) - forward_vector(minus)) \
            / (2.0 * step)
    return jacobian


def unit_covector(direction: np.ndarray, gamma_inverse: np.ndarray) -> np.ndarray:
    return direction / math.sqrt(direction @ gamma_inverse @ direction)


def audit_state(regular_vector: np.ndarray, directions: Iterable[np.ndarray]):
    parent = forward_vector(regular_vector)
    J = forward_jacobian(regular_vector)
    H_parent = parent_symmetrizer(parent)
    H_regular = J.T @ H_parent @ J
    parent_matrices = parent_principal_matrices(parent)
    _, _, gamma_inverse, _, _ = parent_geometry(parent)

    eigenvalues_H = np.linalg.eigvalsh(H_regular)
    minimum_H = float(eigenvalues_H[0])
    maximum_H = float(eigenvalues_H[-1])
    condition_H = float(np.linalg.cond(H_regular))
    diagonal_scale = np.diag(1.0 / np.sqrt(np.diag(H_regular)))
    equilibrated_H = diagonal_scale @ H_regular @ diagonal_scale
    equilibrated_eigenvalues = np.linalg.eigvalsh(equilibrated_H)
    minimum_equilibrated_H = float(equilibrated_eigenvalues[0])
    condition_equilibrated_H = float(np.linalg.cond(equilibrated_H))
    worst_symmetry = 0.0
    worst_imaginary = 0.0
    worst_eigenvector_condition = 0.0
    for coordinate_direction in directions:
        normal = unit_covector(np.asarray(coordinate_direction), gamma_inverse)
        parent_normal = sum(normal[k] * parent_matrices[k] for k in range(3))
        regular_normal = np.linalg.solve(J, parent_normal @ J)
        product = H_regular @ regular_normal
        scale = max(1.0, float(np.max(np.abs(product))))
        worst_symmetry = max(
            worst_symmetry,
            float(np.max(np.abs(product - product.T))) / scale)
        eigenvalues = np.linalg.eigvals(regular_normal)
        worst_imaginary = max(worst_imaginary,
                              float(np.max(np.abs(eigenvalues.imag))))

        # Diagonal congruence removes unit/power scaling before constructing an
        # H-orthonormal basis.  Whether this equilibration has a regular
        # continuum interpretation at the puncture is audited separately.
        equilibrated_normal = np.linalg.solve(
            diagonal_scale, regular_normal @ diagonal_scale)
        values, vectors = np.linalg.eigh(equilibrated_H)
        if values[0] <= 0.0:
            worst_eigenvector_condition = math.inf
            continue
        H_half = (vectors * np.sqrt(values)) @ vectors.T
        H_minus_half = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
        symmetric_normal = H_half @ equilibrated_normal @ H_minus_half
        symmetric_normal = 0.5 * (symmetric_normal + symmetric_normal.T)
        _, orthogonal = np.linalg.eigh(symmetric_normal)
        right_vectors = H_minus_half @ orthogonal
        worst_eigenvector_condition = max(
            worst_eigenvector_condition, float(np.linalg.cond(right_vectors)))
    return {
        "minimum_H": minimum_H,
        "maximum_H": maximum_H,
        "condition_H": condition_H,
        "minimum_equilibrated_H": minimum_equilibrated_H,
        "condition_equilibrated_H": condition_equilibrated_H,
        "condition_J": float(np.linalg.cond(J)),
        "symmetry_residual": worst_symmetry,
        "maximum_imaginary_eigenvalue": worst_imaginary,
        "eigenvector_condition": worst_eigenvector_condition,
    }


def minkowski_state() -> np.ndarray:
    vector = np.zeros(58)
    vector[:5] = [1.0, 0.0, 0.0, 1.0, 0.0]
    vector[5:7] = 1.0
    return vector


def weak_random_state(rng: np.random.Generator) -> np.ndarray:
    vector = minkowski_state()
    vector[:5] += rng.uniform(-0.02, 0.02, 5)
    vector[5:7] += rng.uniform(-0.02, 0.02, 2)
    vector[7:] = rng.uniform(-0.02, 0.02, 51)
    return vector


def wormhole_state(radius: float) -> np.ndarray:
    psi = 1.0 + 0.5 / radius
    chi = psi ** -4
    vector = minkowski_state()
    vector[5] = chi
    vector[6] = chi
    # X=Y=d(psi^-4)/dx along the x-axis.
    gradient = 2.0 * psi ** -5 / radius ** 2
    vector[35:38] = [gradient, 0.0, 0.0]
    vector[38:41] = [gradient, 0.0, 0.0]
    return vector


def parent_characteristic_matrix_x(parent_vector: np.ndarray,
                                   gamma2: float = 1.0) -> np.ndarray:
    """Rows are Eqs. (B6)--(B10) for an x-directed physical unit normal."""
    _, _, gamma_inverse, beta, _ = parent_geometry(parent_vector)
    normal_down = np.array([1.0 / math.sqrt(gamma_inverse[0, 0]), 0.0, 0.0])
    normal_up = gamma_inverse @ normal_down
    normal_spacetime = np.concatenate(([beta @ normal_down], normal_down))
    matrix = np.zeros((58, 58))
    row = 0
    matrix[row:row + 10, :10] = np.eye(10)
    row += 10
    for sign in (1.0, -1.0):
        for q, (a, b) in enumerate(PAIRS):
            matrix[row, 10 + q] = 1.0
            matrix[row, q] = -gamma2
            for i in range(3):
                matrix[row, 20 + 10 * i + q] = sign * normal_up[i]
            for c in range(4):
                matrix[row, 50 + c] += sign * (
                    normal_spacetime[a] * float(b == c)
                    + normal_spacetime[b] * float(a == c))
            row += 1
    # For this x-directed normal on the conformally flat wormhole, y and z
    # give an independent basis for P_i^k Phi_kab.
    for i in (1, 2):
        matrix[row:row + 10, 20 + 10 * i:30 + 10 * i] = np.eye(10)
        row += 10
    matrix[row:row + 4, 50:54] = np.eye(4)
    row += 4
    matrix[row:row + 4, 54:58] = np.eye(4)
    row += 4
    if row != 58:
        raise AssertionError(row)
    return matrix


def characteristic_subspace_condition(regular_vector: np.ndarray,
                                      relative_step: float = 2.0e-6) -> float:
    """Amplitude- and repeated-zero-eigenspace-independent kappa(R)."""
    parent = forward_vector(regular_vector)
    J = forward_jacobian(regular_vector, relative_step)
    characteristic = parent_characteristic_matrix_x(parent)
    right = np.linalg.solve(J, np.linalg.inv(characteristic))

    # Columns 0:10, 30:58 share zero speed for gamma1=-1.  Columns 10:20 and
    # 20:30 are the repeated + and - wave eigenspaces. Arbitrary mixing inside
    # any repeated eigenspace must not be mistaken for ill-conditioning, so
    # replace all three groups by orthonormal bases for the same subspaces.
    zero_columns = list(range(10)) + list(range(30, 58))
    zero_basis = np.linalg.qr(right[:, zero_columns])[0]
    plus_basis = np.linalg.qr(right[:, 10:20])[0]
    minus_basis = np.linalg.qr(right[:, 20:30])[0]
    balanced = np.column_stack((zero_basis, plus_basis, minus_basis))
    return float(np.linalg.cond(balanced))


def wormhole_conditioning_scan():
    rows = []
    for radius in (0.5, 0.25, 0.125, 0.0625):
        state = wormhole_state(radius)
        primary = characteristic_subspace_condition(state, 1.0e-5)
        repeat = characteristic_subspace_condition(state, 3.0e-6)
        rows.append((radius, primary, repeat))
    slope, _ = np.polyfit(
        np.log([row[0] for row in rows]),
        np.log([math.sqrt(row[1] * row[2]) for row in rows]), 1)
    return rows, float(slope)


def sample_directions(rng: np.random.Generator, count: int):
    directions = [np.eye(3)[i] for i in range(3)]
    for _ in range(count):
        direction = rng.normal(size=3)
        directions.append(direction / np.linalg.norm(direction))
    return directions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-states", type=int, default=8)
    parser.add_argument("--directions", type=int, default=16)
    args = parser.parse_args()
    rng = np.random.default_rng(20260818)
    directions = sample_directions(rng, args.directions)

    cases = [("minkowski", minkowski_state())]
    cases.extend((f"weak_random_{i}", weak_random_state(rng))
                 for i in range(args.random_states))
    cases.extend((f"wormhole_r_{radius:g}", wormhole_state(radius))
                 for radius in (0.5, 0.25))

    worst_symmetry = 0.0
    worst_imaginary = 0.0
    for name, state in cases:
        result = audit_state(state, directions)
        worst_symmetry = max(worst_symmetry, result["symmetry_residual"])
        worst_imaginary = max(worst_imaginary,
                              result["maximum_imaginary_eigenvalue"])
        print(name + " " + " ".join(
            f"{key}={value:.17e}" for key, value in result.items()))
    if worst_symmetry > 2.0e-6:
        raise AssertionError(f"transformed symmetrizer residual: {worst_symmetry}")
    if worst_imaginary > 2.0e-7:
        raise AssertionError(f"complex characteristic speed: {worst_imaginary}")
    print("MATCHED_FINITE_RADIUS_HYPERBOLICITY=PASS")
    conditioning, slope = wormhole_conditioning_scan()
    for radius, primary, repeat in conditioning:
        print(f"wormhole_characteristic_condition r={radius:.8e} "
              f"step1e-5={primary:.17e} step3e-6={repeat:.17e}")
    print(f"wormhole_characteristic_condition_log_slope={slope:.17e}")
    if max(max(row[1], row[2]) for row in conditioning) > 1.0e10 \
            and slope < -2.0:
        print("FORMULATION NOT ESTABLISHED: characteristic subspaces become "
              "uncontrollably tangent as the wormhole puncture is refined")
        return 2
    print("PUNCTURE_CONDITIONING=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
