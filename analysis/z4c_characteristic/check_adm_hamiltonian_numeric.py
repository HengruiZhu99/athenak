"""Analytic and coordinate-invariance checks for the independent ADM diagnostic.

Run with a Python environment providing NumPy. This does not launch AthenaK.
"""

import json

import numpy as np

from adm_hamiltonian import hamiltonian, ricci_tensor


def main():
    rng = np.random.default_rng(407)
    xyz = rng.normal(size=(200, 3))
    # g_ij=exp(2f) delta_ij, f=a r^2: R=exp(-2f)(-4 Lap f-2|grad f|^2).
    a, k = 0.07, 0.13
    f = a * np.sum(xyz**2, axis=1)
    df = 2 * a * xyz
    ddf = 2 * a * np.eye(3)[None, :, :]
    g = np.exp(2 * f)[:, None, None] * np.eye(3)[None, :, :]
    dg = 2 * df[:, :, None, None] * g[:, None, :, :]
    ddg = (2 * ddf + 4 * df[:, :, None] * df[:, None, :])[:, :, :, None, None]
    ddg = ddg * g[:, None, None, :, :]
    extrinsic = k * g
    expected = np.exp(-2 * f) * (-24 * a - 2 * np.sum(df**2, axis=1)) + 6 * k * k
    inputs = [q.copy() for q in (g, dg, ddg, extrinsic)]
    actual = hamiltonian(g, dg, ddg, extrinsic)
    analytic_error = float(np.max(abs(actual - expected)))
    assert analytic_error < 1e-13, analytic_error
    expected_ricci = (df[:, :, None] * df[:, None, :]
                      - (8*a + np.sum(df**2, axis=1))[:, None, None] * np.eye(3))
    ricci_error = float(np.max(abs(ricci_tensor(g, dg, ddg) - expected_ricci)))
    assert ricci_error < 1e-13, ricci_error
    assert all(np.array_equal(q, old) for q, old in zip((g, dg, ddg, extrinsic), inputs))

    # An affine coordinate change makes the metric non-diagonal and non-isotropic.
    # Transform derivative indices as well as both metric/extrinsic indices.
    transform = np.array([[1.2, 0.2, -0.1], [0.1, 0.8, 0.15], [-0.1, 0.05, 1.1]])
    gt = np.einsum("ia,jb,nij->nab", transform, transform, g)
    dgt = np.einsum("pa,ib,jc,npij->nabc", transform, transform, transform, dg)
    ddgt = np.einsum(
        "pa,qb,ic,jd,npqij->nabcd", transform, transform, transform, transform, ddg
    )
    kt = np.einsum("ia,jb,nij->nab", transform, transform, extrinsic)
    coordinate_error = float(np.max(abs(hamiltonian(gt, dgt, ddgt, kt) - expected)))
    assert coordinate_error < 1e-13, coordinate_error
    transformed_ricci = np.einsum("ia,jb,nij->nab", transform, transform, expected_ricci)
    ricci_coordinate_error = float(np.max(abs(ricci_tensor(gt, dgt, ddgt) - transformed_ricci)))
    assert ricci_coordinate_error < 1e-13, ricci_coordinate_error

    # Flat Cartesian space under X_i=exp(x_i): nonconstant metric but R=0.
    diagonal = np.exp(2 * xyz)
    flat = np.zeros_like(g)
    dflat, ddflat = np.zeros_like(dg), np.zeros_like(ddg)
    for i in range(3):
        flat[:, i, i] = diagonal[:, i]
        dflat[:, i, i, i] = 2 * diagonal[:, i]
        ddflat[:, i, i, i, i] = 4 * diagonal[:, i]
    flat_error = float(np.max(abs(hamiltonian(flat, dflat, ddflat, 0 * flat))))
    assert flat_error < 1e-12, flat_error

    # A complex direction in isotropic K has known derivative dH/dk=12k.
    response = hamiltonian(g, dg, ddg, (k + 1e-30j) * g).imag / 1e-30
    response_error = float(np.max(abs(response - 12 * k)))
    assert response_error < 1e-13, response_error
    # Derivative-only complex inputs must also retain their imaginary parts.
    derivative_response = hamiltonian(g, (1 + 1e-30j) * dg, ddg, extrinsic).imag / 1e-30
    eps = 1e-5
    centered = (hamiltonian(g, (1 + eps) * dg, ddg, extrinsic)
                - hamiltonian(g, (1 - eps) * dg, ddg, extrinsic)) / (2 * eps)
    derivative_response_error = float(np.max(abs(derivative_response - centered)))
    assert derivative_response_error < 1e-9, derivative_response_error

    # Schwarzschild M=1 in ingoing Kerr-Schild coordinates, including points
    # inside the horizon. Supply analytic derivatives, not FD approximations.
    direction = xyz / np.linalg.norm(xyz, axis=1)[:, None]
    radius = np.geomspace(0.4, 10.0, len(xyz))
    x = direction * radius[:, None]
    identity = np.eye(3)
    ks = identity[None, :, :] + 2 * x[:, :, None] * x[:, None, :] / radius[:, None, None]**3
    dks, ddks = np.zeros_like(dg), np.zeros_like(ddg)
    for a in range(3):
        for i in range(3):
            for j in range(3):
                dks[:, a, i, j] = (
                    2 * (identity[i, a] * x[:, j] + x[:, i] * identity[j, a]) / radius**3
                    - 6 * x[:, i] * x[:, j] * x[:, a] / radius**5
                )
                for b in range(3):
                    ddks[:, a, b, i, j] = (
                        2 * (identity[i, a] * identity[j, b] + identity[i, b] * identity[j, a]) / radius**3
                        - 6 * ((identity[i, a] * x[:, j] + x[:, i] * identity[j, a]) * x[:, b]
                               + (identity[i, b] * x[:, j] + x[:, i] * identity[j, b]) * x[:, a]
                               + x[:, i] * x[:, j] * identity[a, b]) / radius**5
                        + 30 * x[:, i] * x[:, j] * x[:, a] * x[:, b] / radius**7
                    )
    kks = 2 / (radius**2 * np.sqrt(1 + 2 / radius))[:, None, None] * (
        identity[None, :, :] - (2 + 1 / radius)[:, None, None]
        * direction[:, :, None] * direction[:, None, :]
    )
    schwarzschild_error = float(np.max(abs(hamiltonian(ks, dks, ddks, kks))))
    assert schwarzschild_error < 3e-12, schwarzschild_error
    report = {
        "samples_per_case": len(xyz),
        "analytic_conformal_metric_max_error": analytic_error,
        "analytic_Ricci_tensor_max_error": ricci_error,
        "affine_coordinate_change_max_error": coordinate_error,
        "affine_Ricci_tensor_change_max_error": ricci_coordinate_error,
        "flat_curvilinear_metric_max_error": flat_error,
        "complex_direction_max_error": response_error,
        "complex_metric_derivative_vs_centered_error": derivative_response_error,
        "Schwarzschild_analytic_vacuum_max_error": schwarzschild_error,
        "inputs_unchanged": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
