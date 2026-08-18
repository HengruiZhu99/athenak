#!/usr/bin/env python3
"""Independent numerical audit of the stationary trumpet radial provider."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import brentq

try:
    from . import generate_trumpet_table as trumpet
    from .reference_frame_audit import NVAR, principal_matrix, symmetrizer
except ImportError:
    import generate_trumpet_table as trumpet
    from reference_frame_audit import NVAR, principal_matrix, symmetrizer


def interpolate(log_r: np.ndarray, values: np.ndarray, radius: float) -> np.ndarray:
    """Mirror the C++ eight-point log-radius interpolant and its derivatives."""
    spacing = log_r[1] - log_r[0]
    coordinate = (np.log(radius) - log_r[0]) / spacing
    start = max(0, min(len(log_r) - 8, int(np.floor(coordinate)) - 3))
    z = coordinate - start
    weights = np.zeros((3, 8))
    for p in range(8):
        denominator = 1.0
        value = 1.0
        first = 0.0
        second = 0.0
        for q in range(8):
            if q == p:
                continue
            denominator *= p - q
            factor = z - q
            second = second * factor + 2.0 * first
            first = first * factor + value
            value *= factor
        weights[:, p] = (value / denominator,
                         first / (denominator * spacing),
                         second / (denominator * spacing**2))
    log_derivatives = weights @ values[start:start + 8]
    return np.array([log_derivatives[0], log_derivatives[1] / radius,
                     (log_derivatives[2] - log_derivatives[1]) / radius**2])


def analytic_characteristic_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a complete orthonormal 50-field basis and its speed signs."""
    direction = direction / np.linalg.norm(direction)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ direction) > 0.8:
        seed = np.array([0.0, 1.0, 0.0])
    tangent1 = seed - (seed @ direction) * direction
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(direction, tangent1)
    basis = np.zeros((NVAR, NVAR))
    speed_sign = np.zeros(NVAR)
    column = 0
    for component in range(10):
        basis[component, column] = 1.0
        speed_sign[column] = 2.0
        column += 1
        for tangent in (tangent1, tangent2):
            for spatial in range(3):
                basis[20 + 10 * spatial + component, column] = tangent[spatial]
            column += 1
        basis[10 + component, column] = 1.0 / np.sqrt(2.0)
        for spatial in range(3):
            basis[20 + 10 * spatial + component, column] = (
                direction[spatial] / np.sqrt(2.0)
            )
        speed_sign[column] = 1.0
        column += 1
        basis[10 + component, column] = 1.0 / np.sqrt(2.0)
        for spatial in range(3):
            basis[20 + 10 * spatial + component, column] = (
                -direction[spatial] / np.sqrt(2.0)
            )
        speed_sign[column] = -1.0
        column += 1
    assert column == NVAR
    return basis, speed_sign


def trumpet_principal_audit(log_r: np.ndarray, data: np.ndarray) -> dict[str, float]:
    """Audit the complete principal basis at the requested closest-grid radii."""
    radii = np.array([1 / 8, 1 / 12, 1 / 16, 1 / 24, 1 / 32, 1 / 48, 1 / 64])
    directions = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0),
    )
    inverse_relative_metric = np.eye(3)
    energy = symmetrizer(inverse_relative_metric)
    maximum_residual = 0.0
    maximum_symmetrizer_residual = 0.0
    maximum_basis_condition = 0.0
    minimum_rank = NVAR
    maximum_imaginary_eigenvalue = 0.0
    for radius in radii:
        alpha = interpolate(log_r, data[0], radius)[0]
        psi2 = interpolate(log_r, data[3], radius)[0]
        shift_q = interpolate(log_r, data[6], radius)[0]
        beta_ref = np.array([psi2 * shift_q * radius, 0.0, 0.0])
        for direction in directions:
            matrix = principal_matrix(
                alpha, beta_ref, inverse_relative_metric, direction
            )
            basis, signs = analytic_characteristic_basis(direction)
            beta_s = beta_ref @ direction
            speeds = np.full(NVAR, -beta_s)
            speeds[signs == 2.0] = 0.0
            speeds[signs == 1.0] += alpha
            speeds[signs < 0.0] -= alpha
            maximum_residual = max(
                maximum_residual,
                float(np.max(np.abs(matrix @ basis - basis * speeds))),
            )
            maximum_symmetrizer_residual = max(
                maximum_symmetrizer_residual,
                float(np.max(np.abs(energy @ matrix - (energy @ matrix).T))),
            )
            maximum_basis_condition = max(
                maximum_basis_condition, float(np.linalg.cond(basis))
            )
            minimum_rank = min(minimum_rank, int(np.linalg.matrix_rank(basis)))
            maximum_imaginary_eigenvalue = max(
                maximum_imaginary_eigenvalue,
                float(np.max(np.abs(np.linalg.eigvals(matrix).imag))),
            )
    return {
        "trumpet_principal_basis_residual": maximum_residual,
        "trumpet_principal_symmetrizer_residual": maximum_symmetrizer_residual,
        "trumpet_principal_basis_condition": maximum_basis_condition,
        "trumpet_principal_minimum_rank": float(minimum_rank),
        "trumpet_principal_max_imaginary_eigenvalue": maximum_imaginary_eigenvalue,
    }


def main() -> None:
    log_r, data, table_residual, _, _ = trumpet.build_table(1025, 1.0e-4, 128.0)
    rng = np.random.default_rng(0x1A2B3C)
    sample_radii = np.exp(rng.uniform(np.log(1.0 / 64.0), np.log(64.0), 256))
    value_error = 0.0
    implicit_error = 0.0
    first_derivative_error = 0.0
    second_derivative_error = 0.0
    for radius in sample_radii:
        alpha = interpolate(log_r, data[0], radius)
        psi2 = interpolate(log_r, data[3], radius)
        shift_q = interpolate(log_r, data[6], radius)
        areal = psi2[0] * radius
        alpha_r, alpha_rr = trumpet.alpha_radius_derivatives(
            np.array([alpha[0]]), np.array([areal]))
        areal_d1 = alpha[0] * areal / radius
        expected_alpha_d1 = alpha_r[0] * areal_d1
        areal_d2 = ((expected_alpha_d1 * areal + alpha[0] * areal_d1) / radius
                    - alpha[0] * areal / radius**2)
        expected_alpha_d2 = alpha_rr[0] * areal_d1**2 + alpha_r[0] * areal_d2
        expected_psi2_d1 = areal_d1 / radius - areal / radius**2
        expected_psi2_d2 = (areal_d2 / radius - 2.0 * areal_d1 / radius**2
                            + 2.0 * areal / radius**3)
        log_q_d1 = expected_alpha_d1 / trumpet.N - 3.0 * areal_d1 / areal
        log_q_d2 = (expected_alpha_d2 / trumpet.N
                    - 3.0 * (areal_d2 / areal - (areal_d1 / areal)**2))
        expected_q_d1 = shift_q[0] * log_q_d1
        expected_q_d2 = shift_q[0] * (log_q_d1**2 + log_q_d2)
        first_derivative_error = max(
            first_derivative_error,
            abs(alpha[1] - expected_alpha_d1),
            abs(psi2[1] - expected_psi2_d1) / max(1.0, abs(expected_psi2_d1)),
            abs(shift_q[1] - expected_q_d1),
        )
        second_derivative_error = max(
            second_derivative_error,
            abs(alpha[2] - expected_alpha_d2) / max(1.0, abs(expected_alpha_d2)),
            abs(psi2[2] - expected_psi2_d2) / max(1.0, abs(expected_psi2_d2)),
            abs(shift_q[2] - expected_q_d2) / max(1.0, abs(expected_q_d2)),
        )
        implicit_error = max(
            implicit_error,
            abs(trumpet.implicit(alpha[0], areal)) / max(1.0, areal**4),
        )

    # A smaller direct set exercises the split-integral normalization independently
    # of the ODE used to populate the committed table.
    for radius in np.exp(rng.uniform(np.log(1.0 / 64.0), np.log(64.0), 32)):
        exact_alpha = brentq(
            lambda lapse: trumpet.isotropic_radius_from_alpha(lapse) - radius,
            0.0, 1.0 - 1.0e-12, xtol=1.0e-14,
            rtol=4.0 * np.finfo(float).eps,
        )
        exact_areal = trumpet.radius_from_alpha(exact_alpha)
        exact_q = (np.sqrt(trumpet.C_SQUARED) * np.exp(exact_alpha / trumpet.N)
                   / exact_areal**3)
        value_error = max(
            value_error,
            abs(interpolate(log_r, data[0], radius)[0] - exact_alpha),
            abs(interpolate(log_r, data[3], radius)[0] - exact_areal / radius)
                / max(1.0, exact_areal / radius),
            abs(interpolate(log_r, data[6], radius)[0] - exact_q),
        )

    results = {
        "critical_lapse": trumpet.ALPHA_C,
        "critical_areal_radius": trumpet.RADIUS_C,
        "c_squared": trumpet.C_SQUARED,
        "limiting_areal_radius": trumpet.radius_from_alpha(0.0),
        "table_implicit_residual": table_residual,
        "interpolation_value_error": value_error,
        "interpolation_implicit_error": implicit_error,
        "interpolation_first_derivative_error": first_derivative_error,
        "interpolation_second_derivative_error": second_derivative_error,
        "minimum_tabulated_lapse": float(np.min(data[0])),
        "lapse_monotone": bool(np.all(np.diff(data[0]) > 0.0)),
    }
    results.update(trumpet_principal_audit(log_r, data))
    print(json.dumps(results, indent=2, sort_keys=True))
    assert results["lapse_monotone"]
    assert results["minimum_tabulated_lapse"] > 0.0
    assert table_residual < 2.0e-11
    assert value_error < 2.0e-10
    assert implicit_error < 2.0e-10
    assert first_derivative_error < 2.0e-8
    assert second_derivative_error < 2.0e-5
    assert results["trumpet_principal_basis_residual"] < 1.0e-13
    assert results["trumpet_principal_symmetrizer_residual"] < 1.0e-13
    assert results["trumpet_principal_basis_condition"] < 10.0
    assert results["trumpet_principal_minimum_rank"] == NVAR
    assert results["trumpet_principal_max_imaginary_eigenvalue"] < 1.0e-13


if __name__ == "__main__":
    main()
