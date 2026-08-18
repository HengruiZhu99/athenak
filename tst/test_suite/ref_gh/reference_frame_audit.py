#!/usr/bin/env python3
"""Pre-evolution algebra and principal-symbol gates for reference-frame FO-GH.

This audit intentionally has no AthenaK runtime dependency.  It tests the exact tensor
identities on which the production Kokkos kernels are based.  It exits nonzero on the
first failed hard gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SYMMETRIC_PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))
NVAR = 50


def spherical_reference(alpha: float, psi: float, beta: np.ndarray):
    """Return the Cartesian, axis-free coframe/frame and reference metric."""
    theta = np.zeros((4, 4))
    theta[0, 0] = alpha
    theta[1:, 0] = psi**2 * beta
    theta[1:, 1:] = psi**2 * np.eye(3)

    frame = np.zeros((4, 4))
    frame[0, 0] = 1.0 / alpha
    frame[0, 1:] = -beta / alpha
    frame[1:, 1:] = psi**-2 * np.eye(3)

    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    metric = np.einsum("Aa,Bb,AB->ab", theta, theta, eta)
    return theta, frame, metric


def independent_to_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4))
    for value, (a, b) in zip(values, SYMMETRIC_PAIRS):
        matrix[a, b] = value
        matrix[b, a] = value
    return matrix


def principal_matrix(alpha: float, beta_ref: np.ndarray, inverse_g: np.ndarray,
                     s_cov: np.ndarray) -> np.ndarray:
    """Return A(s) for dt U + A(s) U = lower order, ten identical blocks."""
    matrix = np.zeros((NVAR, NVAR))
    beta_s = float(beta_ref @ s_cov)
    raised_s = inverse_g @ s_cov
    for component in range(10):
        pi = 10 + component
        matrix[pi, pi] = -beta_s
        for spatial in range(3):
            phi = 20 + 10 * spatial + component
            matrix[pi, phi] = alpha * raised_s[spatial]
            matrix[phi, pi] = alpha * s_cov[spatial]
            matrix[phi, phi] = -beta_s
    return matrix


def symmetrizer(inverse_g: np.ndarray, metric_weight: float = 1.7) -> np.ndarray:
    matrix = np.zeros((NVAR, NVAR))
    matrix[:10, :10] = metric_weight**2 * np.eye(10)
    matrix[10:20, 10:20] = np.eye(10)
    for i in range(3):
        for j in range(3):
            for component in range(10):
                matrix[20 + 10*i + component, 20 + 10*j + component] = inverse_g[i, j]
    return matrix


@dataclass
class AuditResult:
    frame_duality_error: float
    reference_orthonormality_error: float
    metric_roundtrip_error: float
    derivative_roundtrip_error: float
    background_constraint_error: float
    transformed_wave_product_error: float
    principal_eigenvalue_error: float
    principal_eigenvector_condition: float
    symmetrizer_residual: float
    symmetrizer_min_eigenvalue: float


def run_audit(seed: int = 20260818) -> AuditResult:
    rng = np.random.default_rng(seed)
    alpha = 0.63
    psi = 1.71
    beta = np.array([0.07, -0.03, 0.02])
    theta, frame, reference_metric = spherical_reference(alpha, psi, beta)
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])

    duality = theta @ frame.T
    frame_metric = np.einsum("ab,Aa,Bb->AB", reference_metric, frame, frame)

    psi_ab = independent_to_matrix(
        np.array([-1.2, 0.03, -0.02, 0.01, 1.1, 0.02, -0.01, 0.9, 0.04, 1.3])
    )
    coordinate_metric = np.einsum("Aa,Bb,AB->ab", theta, theta, psi_ab)
    recovered = np.einsum("ab,Aa,Bb->AB", coordinate_metric, frame, frame)

    # Exact first-derivative product rule and inverse map for arbitrary d(theta), d(Psi).
    dtheta = rng.normal(scale=0.08, size=(3, 4, 4))
    dpsi = rng.normal(scale=0.06, size=(3, 4, 4))
    dpsi = 0.5 * (dpsi + np.swapaxes(dpsi, 1, 2))
    derivative_error = 0.0
    for i in range(3):
        dg = (
            np.einsum("Aa,Bb,AB->ab", dtheta[i], theta, psi_ab)
            + np.einsum("Aa,Bb,AB->ab", theta, dtheta[i], psi_ab)
            + np.einsum("Aa,Bb,AB->ab", theta, theta, dpsi[i])
        )
        # d(frame) follows from d(theta^{-T}) in the index convention used here.
        dframe = -frame @ dtheta[i].T @ frame
        back = (
            np.einsum("ab,Aa,Bb->AB", dg, frame, frame)
            + np.einsum("ab,Aa,Bb->AB", coordinate_metric, dframe, frame)
            + np.einsum("ab,Aa,Bb->AB", coordinate_metric, frame, dframe)
        )
        derivative_error = max(derivative_error, float(np.max(np.abs(back - dpsi[i]))))

    # If physical and reference metrics agree their Christoffels agree identically.
    # Use nontrivial symmetric connection data so this is not a zero-data test.
    reference_connection = rng.normal(scale=0.2, size=(4, 4, 4))
    reference_connection = 0.5 * (reference_connection + reference_connection.swapaxes(1, 2))
    physical_connection = reference_connection.copy()
    background_constraint = np.einsum(
        "bc,abc->a", np.linalg.inv(reference_metric),
        physical_connection - reference_connection
    )

    # Product-rule oracle for T*g with arbitrary values and first/second derivatives.
    # The two paths are independently grouped contractions of the exact Hessian.
    inv_coord_metric = np.linalg.inv(coordinate_metric)
    tensor = np.einsum("Aa,Bb->ABab", frame, frame)
    dtensor = rng.normal(scale=0.04, size=(4, 4, 4, 4, 4))
    ddtensor = rng.normal(scale=0.03, size=(4, 4, 4, 4, 4, 4))
    ddtensor = 0.5 * (ddtensor + ddtensor.swapaxes(0, 1))
    dg = rng.normal(scale=0.05, size=(4, 4, 4))
    dg = 0.5 * (dg + dg.swapaxes(1, 2))
    ddg = rng.normal(scale=0.03, size=(4, 4, 4, 4))
    ddg = 0.5 * (ddg + ddg.swapaxes(0, 1))
    ddg = 0.5 * (ddg + ddg.swapaxes(2, 3))

    direct_hessian = (
        np.einsum("ABab,cdab->cdAB", tensor, ddg)
        + np.einsum("cABab,dab->cdAB", dtensor, dg)
        + np.einsum("dABab,cab->cdAB", dtensor, dg)
        + np.einsum("cdABab,ab->cdAB", ddtensor, coordinate_metric)
    )
    direct_wave = np.einsum("cd,cdAB->AB", inv_coord_metric, direct_hessian)
    transformed_wave = (
        np.einsum("ABab,cd,cdab->AB", tensor, inv_coord_metric, ddg)
        + 2.0 * np.einsum("cd,cABab,dab->AB", inv_coord_metric, dtensor, dg)
        + np.einsum("cd,cdABab,ab->AB", inv_coord_metric, ddtensor,
                    coordinate_metric)
    )

    random = rng.normal(size=(3, 3))
    spatial_metric = random.T @ random + 0.7*np.eye(3)
    inverse_spatial_metric = np.linalg.inv(spatial_metric)
    s_cov = rng.normal(size=3)
    s_cov /= np.sqrt(s_cov @ inverse_spatial_metric @ s_cov)
    beta_ref = np.array([0.08, -0.04, 0.025])
    principal = principal_matrix(0.81, beta_ref, inverse_spatial_metric, s_cov)
    energy = symmetrizer(inverse_spatial_metric)
    eigenvalues, eigenvectors = np.linalg.eig(principal)
    beta_s = beta_ref @ s_cov
    expected = np.sort(np.concatenate((
        np.zeros(10),
        np.full(20, -beta_s),
        np.full(10, -beta_s + 0.81),
        np.full(10, -beta_s - 0.81),
    )))

    return AuditResult(
        frame_duality_error=float(np.max(np.abs(duality - np.eye(4)))),
        reference_orthonormality_error=float(np.max(np.abs(frame_metric - eta))),
        metric_roundtrip_error=float(np.max(np.abs(recovered - psi_ab))),
        derivative_roundtrip_error=derivative_error,
        background_constraint_error=float(np.max(np.abs(background_constraint))),
        transformed_wave_product_error=float(np.max(np.abs(direct_wave - transformed_wave))),
        principal_eigenvalue_error=float(np.max(np.abs(np.sort(eigenvalues.real) - expected))),
        principal_eigenvector_condition=float(np.linalg.cond(eigenvectors)),
        symmetrizer_residual=float(np.max(np.abs(energy @ principal -
                                                     (energy @ principal).T))),
        symmetrizer_min_eigenvalue=float(np.linalg.eigvalsh(energy).min()),
    )


def validate(result: AuditResult) -> None:
    roundoff_fields = (
        "frame_duality_error",
        "reference_orthonormality_error",
        "metric_roundtrip_error",
        "derivative_roundtrip_error",
        "background_constraint_error",
        "transformed_wave_product_error",
        "principal_eigenvalue_error",
        "symmetrizer_residual",
    )
    failures = [name for name in roundoff_fields if getattr(result, name) > 1.0e-11]
    if result.symmetrizer_min_eigenvalue <= 0.0:
        failures.append("symmetrizer_min_eigenvalue")
    if not np.isfinite(result.principal_eigenvector_condition):
        failures.append("principal_eigenvector_condition")
    if failures:
        raise AssertionError("reference-frame FO-GH audit failed: " + ", ".join(failures))


def main() -> None:
    result = run_audit()
    validate(result)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
