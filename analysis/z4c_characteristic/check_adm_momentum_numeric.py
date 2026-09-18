"""Analytic, tensor-transformation, and complex-response ADM momentum checks."""

import json

import numpy as np

from adm_momentum import momentum


def main():
    rng = np.random.default_rng(9134)
    x = rng.normal(size=(200, 3))
    f, df = 0.07 * np.sum(x*x, axis=1), 0.14 * x
    metric = np.exp(2*f)[:, None, None] * np.eye(3)
    first = 2 * df[:, :, None, None] * metric[:, None]
    # K_ij = k(x) g_ij gives M_i = -2 partial_i k for any metric.
    k, dk = 0.13 + 0.04*np.sum(x*x, axis=1), 0.08*x
    extrinsic = k[:, None, None]*metric
    first_extrinsic = (dk[:, :, None, None]*metric[:, None]
                       + k[:, None, None, None]*first)
    inputs = (metric, first, extrinsic, first_extrinsic)
    copies = [q.copy() for q in inputs]
    expected = -2*dk
    analytic = np.max(abs(momentum(*inputs)-expected))
    assert analytic < 1e-13, analytic
    assert all(np.array_equal(q, old) for q, old in zip(inputs, copies))

    transform = np.array([[1.2, .2, -.1], [.1, .8, .15], [-.1, .05, 1.1]])
    def cov2(q):
        return np.einsum("ia,jb,nij->nab", transform, transform, q)
    def cov3(q):
        return np.einsum("pa,ib,jc,npij->nabc", transform, transform, transform, q)
    transformed = momentum(cov2(metric), cov3(first), cov2(extrinsic), cov3(first_extrinsic))
    affine = np.max(abs(transformed-expected@transform))
    assert affine < 1e-13, affine

    # Schwarzschild in ingoing Kerr-Schild coordinates: analytic g and K,
    # their spatial derivatives evaluated by the complex-step chain rule.
    direction = x/np.linalg.norm(x, axis=1)[:, None]
    x = direction*np.geomspace(.4, 10., len(x))[:, None]
    def schwarzschild(points):
        radius = np.sqrt(np.sum(points*points, axis=1))
        n = points/radius[:, None]
        nn = n[:, :, None]*n[:, None, :]
        g = np.eye(3) + (2/radius)[:, None, None]*nn
        K = (2/(radius**2*np.sqrt(1+2/radius)))[:, None, None]*(
            np.eye(3)-(2+1/radius)[:, None, None]*nn)
        return g, K
    g, K = schwarzschild(x)
    dg, dK = np.empty((len(x), 3, 3, 3)), np.empty((len(x), 3, 3, 3))
    for a in range(3):
        z = x.astype(complex)
        z[:, a] += 1e-30j
        gz, Kz = schwarzschild(z)
        dg[:, a], dK[:, a] = gz.imag/1e-30, Kz.imag/1e-30
    vacuum = np.max(abs(momentum(g, dg, K, dK)))
    assert vacuum < 1e-12, vacuum

    # Only derivative input is complex: ensure its imaginary part survives.
    response = momentum(metric, first, extrinsic,
                        first_extrinsic+1e-30j*dk[:, :, None, None]*metric[:, None]).imag/1e-30
    response_error = np.max(abs(response-expected))
    assert response_error < 1e-13, response_error
    # Independent conservative-coordinate identity for arbitrary symmetric jets.
    basis = rng.normal(size=metric.shape)
    g = basis @ np.swapaxes(basis, -1, -2) + np.eye(3)
    dg = rng.normal(size=first.shape)
    dg = (dg + np.swapaxes(dg, -1, -2))/2
    K = rng.normal(size=extrinsic.shape)
    K = (K + np.swapaxes(K, -1, -2))/2
    dK = rng.normal(size=first_extrinsic.shape)
    dK = (dK+np.swapaxes(dK,-1,-2))/2
    inv = np.linalg.inv(g)
    dinv = -np.einsum('nip,napq,nqj->naij', inv, dg, inv)
    # M_i = 1/sqrt(g) partial_j(sqrt(g) K^j_i)
    #       - (1/2) K^jk partial_i g_jk - partial_i K.
    reference = np.einsum('njjk,nki->ni', dinv, K)
    reference += np.einsum('njk,njki->ni', inv, dK)
    reference += .5*np.einsum('nkl,njkl,njm,nmi->ni', inv, dg, inv, K)
    reference -= .5*np.einsum('njk,nijk->ni', inv@K@inv, dg)
    reference -= np.einsum('nijk,njk->ni', dinv, K)+np.einsum('njk,nijk->ni', inv, dK)
    independent = np.max(abs(momentum(g, dg, K, dK)-reference))
    assert independent < 1e-12, independent
    print(json.dumps({"samples_per_case": len(x), "analytic_max_error": float(analytic),
                     "affine_covector_max_error": float(affine),
                     "Schwarzschild_vacuum_max_error": float(vacuum),
                     "complex_derivative_max_error": float(response_error),
                     "independent_coordinate_identity_max_error": float(independent),
                     "inputs_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
