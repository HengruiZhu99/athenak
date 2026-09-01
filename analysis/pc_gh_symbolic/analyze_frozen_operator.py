#!/usr/bin/env python3
"""Analyze matrices extracted directly from the production PC-GH RHS kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


NSTATE = 55
NAMES = (
    "chi", "gtxx", "gtxy", "gtxz", "gtyy", "gtyz", "gtzz", "K",
    "Atxx", "Atxy", "Atxz", "Atyy", "Atyz", "Atzz",
    "Lamx", "Lamy", "Lamz", "pi", "A", "betax", "betay", "betaz",
    "X1", "X2", "X3",
    "Q1xx", "Q1xy", "Q1xz", "Q1yy", "Q1yz", "Q1zz",
    "Q2xx", "Q2xy", "Q2xz", "Q2yy", "Q2yz", "Q2zz",
    "Q3xx", "Q3xy", "Q3xz", "Q3yy", "Q3yz", "Q3zz",
    "Y1", "Y2", "Y3",
    "B11", "B12", "B13", "B21", "B22", "B23", "B31", "B32", "B33",
)
assert len(NAMES) == NSTATE


def complex_jacobian(function, point, output_size):
    jacobian = np.zeros((output_size, point.size))
    epsilon = 1.0e-30
    for column in range(point.size):
        perturbed = point.astype(complex)
        perturbed[column] += 1j*epsilon
        jacobian[:, column] = np.imag(function(perturbed))/epsilon
    return jacobian


def algebraic_constraints(state):
    metric = symmetric_matrix(state, 1)
    inverse = np.linalg.inv(metric)
    values = [np.linalg.det(metric) - 1.0,
              np.sum(inverse*symmetric_matrix(state, 8))]
    values.extend(np.sum(inverse*symmetric_matrix(state, 25 + 6*direction))
                  for direction in range(3))
    return np.asarray(values)


def algebraic_constraint_jacobian(state):
    return complex_jacobian(algebraic_constraints, state, 5)


def modified_wave_covector(wave, spacing, spatial_order):
    theta = wave*spacing
    if spatial_order == 2:
        symbol = np.sin(theta)
    elif spatial_order == 4:
        symbol = 4.0*np.sin(theta)/3.0 - np.sin(2.0*theta)/6.0
    elif spatial_order == 6:
        symbol = (3.0*np.sin(theta)/2.0 - 3.0*np.sin(2.0*theta)/10.0
                  + np.sin(3.0*theta)/30.0)
    else:
        raise AssertionError(f"unsupported centered spatial order {spatial_order}")
    return symbol/spacing


def read_extended_matrix(path: Path):
    state = np.zeros(NSTATE)
    background = np.zeros(NSTATE)
    lower = np.zeros((NSTATE, NSTATE))
    derivative = np.zeros((NSTATE, NSTATE))
    constraint_background = np.zeros(8)
    constraint_lower = np.zeros((8, NSTATE))
    constraint_derivative = np.zeros((8, NSTATE))
    metadata = {"wave": None, "spacing": None, "spatial_order": None,
                "has_constraints": False}
    with path.open() as stream:
        for line in stream:
            if line.startswith("# k=("):
                metadata["wave"] = np.fromstring(
                    line.split("# k=(", 1)[1].split(")", 1)[0], sep=",")
                metadata["spatial_order"] = int(line.rsplit("order=", 1)[1])
                continue
            if line.startswith("# dx=("):
                metadata["spacing"] = np.fromstring(
                    line.split("# dx=(", 1)[1].split(")", 1)[0], sep=",")
                continue
            if line.startswith("#") or not line.strip():
                continue
            kind, row_text, column_text, value_text = line.split()
            row, column, value = int(row_text), int(column_text), float(value_text)
            if kind == "S":
                state[row] = value
            elif kind == "R":
                background[row] = value
            elif kind == "B":
                lower[row, column] = value
            elif kind == "D":
                derivative[row, column] = value
            elif kind == "C":
                constraint_background[row] = value
                metadata["has_constraints"] = True
            elif kind == "L":
                constraint_lower[row, column] = value
            elif kind == "F":
                constraint_derivative[row, column] = value
            else:
                raise ValueError(f"unknown matrix row kind {kind!r}")
    return (state, background, lower, derivative, constraint_background,
            constraint_lower, constraint_derivative, metadata)


def read_matrix(path: Path):
    return read_extended_matrix(path)[:4]


def symmetric_matrix(vector, start):
    xx, xy, xz, yy, yz, zz = vector[start:start + 6]
    return np.array(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))


def store_symmetric(vector, start, matrix):
    vector[start:start + 6] = (matrix[0, 0], matrix[0, 1], matrix[0, 2],
                               matrix[1, 1], matrix[1, 2], matrix[2, 2])


def project_state(state):
    result = state.copy()
    metric = symmetric_matrix(state, 1)
    inverse = np.linalg.inv(metric)
    scale = np.cbrt(1.0/np.linalg.det(metric))
    at = symmetric_matrix(state, 8)
    trace_at = np.sum(inverse*at)
    store_symmetric(result, 1, scale*metric)
    store_symmetric(result, 8, at - metric*trace_at/3.0)
    for direction in range(3):
        start = 25 + 6*direction
        q = symmetric_matrix(state, start)
        trace_q = np.sum(inverse*q)
        store_symmetric(result, start, scale*(q - metric*trace_q/3.0))
    return result


def projection_jacobian(state):
    jacobian = np.zeros((NSTATE, NSTATE))
    for column in range(NSTATE):
        epsilon = 1.0e-7*max(1.0, abs(state[column]))
        plus, minus = state.copy(), state.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        jacobian[:, column] = (project_state(plus) - project_state(minus))/(2.0*epsilon)
    return jacobian


def operator_metrics(operator):
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    hermitian_part = (operator + operator.conj().T)/2.0
    commutator = operator.conj().T@operator - operator@operator.conj().T
    norm = np.linalg.norm(operator, ord="fro")
    return (np.max(eigenvalues.real), np.min(eigenvalues.real),
            np.max(np.abs(eigenvalues)), np.linalg.cond(eigenvectors),
            np.linalg.eigvalsh(hermitian_part)[-1],
            np.linalg.norm(commutator, ord="fro")/max(norm*norm, 1.0e-300))


def gauge_a1_feedback(state, mu_l, mu_s):
    feedback = np.zeros((NSTATE, NSTATE))
    chi = state[0]
    x = state[22:25]
    feedback[18, 18] -= mu_l*chi
    for direction in range(3):
        beta = 19 + direction
        y = 43 + direction
        feedback[beta, beta] -= mu_s*chi
        feedback[y, 18] -= mu_l*x[direction]
        feedback[y, y] -= mu_l*chi
        for component in range(3):
            b = 46 + 3*direction + component
            feedback[b, 19 + component] -= mu_s*x[direction]
            feedback[b, b] -= mu_s*chi
    return feedback


def gh_constraints(state):
    metric = symmetric_matrix(state, 1)
    inverse = np.linalg.inv(metric)
    contracted = np.zeros(3, dtype=state.dtype)
    q = [symmetric_matrix(state, 25 + 6*direction) for direction in range(3)]
    for upper in range(3):
        for j in range(3):
            for k in range(3):
                for lower in range(3):
                    contracted[upper] += 0.5*inverse[j, k]*inverse[upper, lower]*(
                        q[j][lower, k] + q[k][lower, j] - q[lower][j, k])
    return np.concatenate(([state[7] + state[17]], contracted - state[14:17]))


def mode_constraint_diagnostics(state, vector, constraint_lower,
                                constraint_derivative, metadata):
    if not metadata["has_constraints"]:
        return None
    if metadata["wave"] is None or metadata["spacing"] is None:
        raise AssertionError("extended frozen matrix lacks wave/spacing metadata")
    wave = modified_wave_covector(
        metadata["wave"], metadata["spacing"], metadata["spatial_order"])
    production = (constraint_lower.astype(complex)
                  + 1j*constraint_derivative)@vector
    algebraic = algebraic_constraint_jacobian(state)@vector
    gh_direct = complex_jacobian(gh_constraints, state, 4)@vector

    red_x = vector[22:25] - 1j*wave*vector[0]
    red_y = vector[43:46] - 1j*wave*vector[18]
    red_q = np.empty((3, 6), dtype=complex)
    for direction in range(3):
        red_q[direction] = (vector[25 + 6*direction:31 + 6*direction]
                            - 1j*wave[direction]*vector[1:7])
    red_b = (vector[46:55].reshape(3, 3)
             - 1j*wave[:, None]*vector[19:22][None, :])

    curl_x, curl_y, curl_q, curl_b = [], [], [], []
    x = vector[22:25]
    y = vector[43:46]
    q = vector[25:43].reshape(3, 6)
    b = vector[46:55].reshape(3, 3)
    for first in range(3):
        for second in range(first + 1, 3):
            curl_x.append(1j*(wave[first]*x[second] - wave[second]*x[first]))
            curl_y.append(1j*(wave[first]*y[second] - wave[second]*y[first]))
            curl_q.extend(1j*(wave[first]*q[second] - wave[second]*q[first]))
            curl_b.extend(1j*(wave[first]*b[second] - wave[second]*b[first]))

    return {
        "modified_wave": wave,
        "production_GH": np.linalg.norm(production[:4]),
        "production_physical": np.linalg.norm(production[4:8]),
        "GH_crosscheck": np.linalg.norm(production[:4] - gh_direct),
        "algebraic": np.linalg.norm(algebraic),
        "reduction_X": np.linalg.norm(red_x),
        "reduction_Q": np.linalg.norm(red_q),
        "reduction_Y": np.linalg.norm(red_y),
        "reduction_B": np.linalg.norm(red_b),
        "curl_X": np.linalg.norm(curl_x),
        "curl_Q": np.linalg.norm(curl_q),
        "curl_Y": np.linalg.norm(curl_y),
        "curl_B": np.linalg.norm(curl_b),
        "physical_state": np.linalg.norm(vector[:14]),
        "gauge_GH_state": np.linalg.norm(vector[14:22]),
        "geometry_gradient_state": np.linalg.norm(vector[22:43]),
        "gauge_gradient_state": np.linalg.norm(vector[43:55]),
    }


def metrics(path: Path, mu_l=0.0, mu_s=0.0):
    state, residual, lower, derivative = read_matrix(path)
    lower = lower + gauge_a1_feedback(state, mu_l, mu_s)
    operator = lower.astype(complex) + 1j*derivative
    raw = operator_metrics(operator)
    projection = projection_jacobian(state)
    left, singular, _ = np.linalg.svd(projection)
    rank = int(np.count_nonzero(singular > 1.0e-7))
    basis = left[:, :rank]
    projected_operator = basis.conj().T@projection@operator@basis
    projected = operator_metrics(projected_operator)
    return {
        "max_background_rhs": np.max(np.abs(residual)),
        "max_real_eigenvalue": raw[0],
        "min_real_eigenvalue": raw[1],
        "spectral_radius": raw[2],
        "eigenvector_condition": raw[3],
        "euclidean_log_norm": raw[4],
        "nonnormality": raw[5],
        "projection_rank": rank,
        "projected_max_real_eigenvalue": projected[0],
        "projected_min_real_eigenvalue": projected[1],
        "projected_spectral_radius": projected[2],
        "projected_eigenvector_condition": projected[3],
        "projected_euclidean_log_norm": projected[4],
        "projected_nonnormality": projected[5],
        "lower_frobenius": np.linalg.norm(lower, ord="fro"),
        "fd_response_frobenius": np.linalg.norm(derivative, ord="fro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrices", type=Path, nargs="+")
    parser.add_argument("--details", action="store_true",
                        help="print dominant components of the rightmost eigenmode")
    parser.add_argument("--mu-l", type=float, default=0.0,
                        help="linearized bounded Gauge-A1 lapse feedback")
    parser.add_argument("--mu-s", type=float, default=0.0,
                        help="linearized bounded Gauge-A1 shift feedback")
    args = parser.parse_args()
    header = ("file", "max|R|", "raw_max_Re", "raw_min_Re", "raw_rho",
              "raw_condV", "raw_mu2", "raw_nonnormality", "rankP",
              "projected_max_Re", "projected_min_Re", "projected_rho",
              "projected_condV", "projected_mu2", "projected_nonnormality",
              "||B||_F", "||D||_F")
    print(" ".join(header))
    for path in args.matrices:
        result = metrics(path, args.mu_l, args.mu_s)
        values = (path, *result.values())
        print(str(values[0]), *(f"{value:.17e}" for value in values[1:]))
        if not all(np.isfinite(value) for value in values[1:]):
            raise AssertionError(f"nonfinite frozen-operator metric for {path}")
        if args.details:
            (state, _, lower, derivative, _, constraint_lower,
             constraint_derivative, metadata) = read_extended_matrix(path)
            lower = lower + gauge_a1_feedback(state, args.mu_l, args.mu_s)
            operator = lower.astype(complex) + 1j*derivative
            projection = projection_jacobian(state)
            left, singular, _ = np.linalg.svd(projection)
            basis = left[:, singular > 1.0e-7]
            operator = basis.conj().T@projection@operator@basis
            eigenvalues, eigenvectors = np.linalg.eig(operator)
            mode = int(np.argmax(eigenvalues.real))
            vector = basis@eigenvectors[:, mode]
            vector /= np.max(np.abs(vector))
            print(f"  rightmost eigenvalue={eigenvalues[mode]:.17e}")
            for component in np.argsort(np.abs(vector))[-12:][::-1]:
                print(f"  {NAMES[component]:8s} |v|={abs(vector[component]):.8e} "
                      f"phase={np.angle(vector[component]):+.8e}")
            diagnostics = mode_constraint_diagnostics(
                state, vector, constraint_lower, constraint_derivative, metadata)
            if diagnostics is None:
                print("  constraint decomposition unavailable in legacy matrix artifact")
            else:
                print("  constraint/sector decomposition (mode max |component| = 1):")
                for name, value in diagnostics.items():
                    if name == "modified_wave":
                        print("    modified_wave=" + ",".join(f"{entry:.8e}" for entry in value))
                    else:
                        print(f"    {name:27s} {value:.8e}")


if __name__ == "__main__":
    main()
