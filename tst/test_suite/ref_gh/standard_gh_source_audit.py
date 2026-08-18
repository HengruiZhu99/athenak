#!/usr/bin/env python3
"""Independent flat-reference oracle for the scalar-wave and Lindblom GH RHS forms."""

from __future__ import annotations

import json

import numpy as np


def christoffels(metric: np.ndarray, dmetric: np.ndarray):
    inverse = np.linalg.inv(metric)
    first = np.empty((4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                first[a, b, c] = 0.5 * (
                    dmetric[b, a, c] + dmetric[c, a, b] - dmetric[a, b, c]
                )
    second = np.einsum("ad,dbc->abc", inverse, first)
    return inverse, first, second


def coordinate_wave_source(metric: np.ndarray, dmetric: np.ndarray,
                           gamma0: float) -> tuple[np.ndarray, dict]:
    inverse, first, second = christoffels(metric, dmetric)
    alpha = (-inverse[0, 0])**-0.5
    beta = alpha**2 * inverse[0, 1:]
    normal_up = np.concatenate(([1.0 / alpha], -beta / alpha))
    normal_down = np.array([-alpha, 0.0, 0.0, 0.0])
    contracted_down = np.einsum("bc,abc->a", inverse, first)
    constraint = contracted_down  # flat reference: H_a=0

    source = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            quadratic = 0.0
            for c in range(4):
                for d in range(4):
                    for e in range(4):
                        for f in range(4):
                            quadratic += 2.0 * inverse[c, d] * inverse[e, f] * (
                                dmetric[e, c, a] * dmetric[f, d, b]
                                - first[a, c, e] * first[b, d, f]
                            )
            damping = 0.0
            for c in range(4):
                projector = (
                    (normal_down[b] if c == a else 0.0)
                    + (normal_down[a] if c == b else 0.0)
                    - metric[a, b] * normal_up[c]
                )
                damping += gamma0 * projector * constraint[c]
            source[a, b] = quadratic + damping
    return source, {
        "inverse": inverse,
        "first": first,
        "second": second,
        "alpha": alpha,
        "beta": beta,
        "normal_up": normal_up,
        "normal_down": normal_down,
        "constraint": constraint,
    }


def direct_lindblom_lower(metric: np.ndarray, pi: np.ndarray, phi: np.ndarray,
                          gamma0: float) -> np.ndarray:
    inverse = np.linalg.inv(metric)
    alpha = (-inverse[0, 0])**-0.5
    beta = alpha**2 * inverse[0, 1:]
    dmetric = np.empty((4, 4, 4))
    dmetric[0] = -alpha * pi + np.einsum("i,iab->ab", beta, phi)
    dmetric[1:] = phi
    _, first, _ = christoffels(metric, dmetric)
    spatial_inverse = np.linalg.inv(metric[1:, 1:])
    normal_up = np.concatenate(([1.0 / alpha], -beta / alpha))
    normal_down = np.array([-alpha, 0.0, 0.0, 0.0])
    constraint = np.einsum("bc,abc->a", inverse, first)
    result = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            value = 0.0
            for c in range(4):
                for d in range(4):
                    inner = -pi[c, a] * pi[d, b]
                    for i in range(3):
                        for j in range(3):
                            inner += spatial_inverse[i, j] * phi[i, c, a] * phi[j, d, b]
                    connection_product = 0.0
                    for e in range(4):
                        for f in range(4):
                            connection_product += inverse[e, f] * first[a, c, e] * first[b, d, f]
                    value += 2.0 * alpha * inverse[c, d] * (inner - connection_product)
            nnpi = np.einsum("c,d,cd", normal_up, normal_up, pi)
            value -= 0.5 * alpha * nnpi * pi[a, b]
            for c in range(4):
                for i in range(3):
                    for j in range(3):
                        value -= alpha * normal_up[c] * pi[c, i + 1] \
                                 * spatial_inverse[i, j] * phi[j, a, b]
            for c in range(4):
                projector = (
                    (normal_down[b] if c == a else 0.0)
                    + (normal_down[a] if c == b else 0.0)
                    - metric[a, b] * normal_up[c]
                )
                value += alpha * gamma0 * projector * constraint[c]
            result[a, b] = value
    return result


def scalar_wave_lower(metric: np.ndarray, pi: np.ndarray, phi: np.ndarray,
                      gamma0: float) -> np.ndarray:
    inverse = np.linalg.inv(metric)
    alpha = (-inverse[0, 0])**-0.5
    beta = alpha**2 * inverse[0, 1:]
    dmetric = np.empty((4, 4, 4))
    dmetric[0] = -alpha * pi + np.einsum("i,iab->ab", beta, phi)
    dmetric[1:] = phi
    partial_source, geometry = coordinate_wave_source(metric, dmetric, gamma0)
    contracted_upper = np.einsum("bc,abc->a", inverse, geometry["second"])
    covariant_source = partial_source - np.einsum("c,cab->ab", contracted_upper, dmetric)

    spatial_metric = metric[1:, 1:]
    spatial_inverse = np.linalg.inv(spatial_metric)
    spatial_connection = np.zeros((3, 3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for ell in range(3):
                    spatial_connection[k, i, j] += 0.5 * spatial_inverse[k, ell] * (
                        phi[i, ell + 1, j + 1] + phi[j, ell + 1, i + 1]
                        - phi[ell, i + 1, j + 1]
                    )
    extrinsic = -alpha * geometry["second"][0, 1:, 1:]
    trace_k = np.einsum("ij,ij", spatial_inverse, extrinsic)
    d_inverse_00 = -np.einsum("a,b,iab->i", inverse[0], inverse[0], phi)
    d_alpha = 0.5 * alpha**3 * d_inverse_00

    connection_divergence = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                connection_divergence += spatial_inverse[i, j] \
                                         * spatial_connection[k, i, j] * phi[k]
    lapse_gradient = np.einsum("ij,i,jab->ab", spatial_inverse, d_alpha, phi)
    # With Box Psi = S and Pi=-n^a partial_a Psi, the source and lapse-gradient
    # signs are +alpha*S and -Phi_i D^i alpha.  These signs are independently
    # fixed by the Minkowski identity Pi_t=S-Laplacian(Psi) and reproduce
    # Lindblom et al. Eq. (36) below to roundoff.
    return alpha * (
        connection_divergence + trace_k * pi + covariant_source
    ) - lapse_gradient


def run_audit(seed: int = 5102093) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # Build a generic Lorentzian metric from ADM data.
    spatial_seed = rng.normal(scale=0.12, size=(3, 3))
    spatial = spatial_seed.T @ spatial_seed + np.diag([1.1, 0.9, 1.2])
    alpha = 0.83
    beta = np.array([0.07, -0.04, 0.03])
    metric = np.zeros((4, 4))
    metric[1:, 1:] = spatial
    metric[0, 1:] = spatial @ beta
    metric[1:, 0] = metric[0, 1:]
    metric[0, 0] = -alpha**2 + beta @ spatial @ beta
    pi = rng.normal(scale=0.04, size=(4, 4))
    pi = 0.5 * (pi + pi.T)
    phi = rng.normal(scale=0.05, size=(3, 4, 4))
    phi = 0.5 * (phi + phi.swapaxes(1, 2))
    direct = direct_lindblom_lower(metric, pi, phi, gamma0=1.3)
    scalar = scalar_wave_lower(metric, pi, phi, gamma0=1.3)
    return {
        "scalar_vs_lindblom_max_error": float(np.max(np.abs(scalar - direct))),
        "scalar_vs_lindblom_l2_error": float(np.linalg.norm(scalar - direct)),
    }


def validate(result: dict[str, float]) -> None:
    if result["scalar_vs_lindblom_max_error"] > 2.0e-11:
        raise AssertionError(f"scalar-wave/Lindblom RHS disagreement: {result}")


def main() -> None:
    result = run_audit()
    validate(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
