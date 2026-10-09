"""Constrained symbol gate extracted from the actual C++ tensor/gauge kernels.

Needs numpy. Runs the supplied hyperboloidal_kernel_symbol executable. The
orthonormal frozen normalization is finite only for alpha, chi > 0 and SPD g.
This test does not establish nonlinear pole closure or discrete stability.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def scalar_matrix(f, mu, ea, ec):
    return np.array([
        [0, 0, 0, -f, 0, 0, 0, 0],
        [0, 0, 0, 2 / 3, 4 / 3, 0, 0, -2 / 3],
        [0, 0, 0, 0, 0, -2, 0, 4 / 3],
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1 / 2, 0],
        [-2 / 3, 1 / 3, -1 / 2, 0, 0, 0, 2 / 3, 0],
        [0, 0, 0, -4 / 3, -2 / 3, 0, 0, 4 / 3],
        [-ea, ec, 0, 0, 0, 0, mu, 0],
    ])


def canceled_basis(alpha, w, q0=0.5):
    f, q, mu = 1 + (1 - w) * 2 / alpha, q0 + (1 - q0) * w, (1 - w) * 3 * q0 / 4 + w
    rows, speeds = [], []
    for sign in [-1, 1]:
        rows += [[0, -sign, -sign / 2, -2 / 3, -4 / 3, 1, 0, 0],
                 [0, 2, 0, 0, 2 * sign, 0, 1, 0]]
        speeds += [sign, sign]
    for sign in [-1, 1]:
        rows += [[-sign / np.sqrt(f), 0, 0, 1, 0, 0, 0, 0]]
        speeds += [sign * np.sqrt(f)]
    excess = 2 / alpha
    for sign in [-1, 1]:
        lam = sign * np.sqrt(q)
        rows += [[-lam / (excess + 1 - q0), lam / (2 * (1 - q0)), 0,
                  (q0 - w * excess) / (excess + 1 - q0), q0 / (2 * (1 - q0)), 0,
                  lam * (1 - 3 * q0 / 4) / (1 - q0), 1]]
        speeds += [lam]
    basis = np.zeros((20, 20))
    basis[:8, :8] = rows
    for start in [8, 12]:
        z = np.sqrt(mu)
        basis[start:start + 4, start:start + 4] = [
            [1 / 2, 1, -1 / 2, 0], [-1 / 2, 1, 1 / 2, 0], [0, 0, -z, 1], [0, 0, z, 1]]
        speeds += [-1, 1, -z, z]
    for start in [16, 18]:
        basis[start:start + 2, start:start + 2] = [[1 / 2, 1], [-1 / 2, 1]]
        speeds += [-1, 1]
    return basis, np.array(speeds)


def main(executable):
    data = json.loads(subprocess.check_output(
        [str(Path(executable).resolve())], text=True))
    maximum_error = maximum_condition = maximum_eigen_error = 0.0
    for case in data:
        matrix = np.array(case['M'])
        expected = np.zeros((20, 20))
        expected[:8, :8] = scalar_matrix(case['f'], case['mu'], case['W'], case['W'] / 2)
        vector = [[0, -2, 0, 1], [-1 / 2, 0, 1 / 2, 0],
                  [0, 0, 0, 1], [0, 0, case['mu'], 0]]
        for start in [8, 12]:
            expected[start:start + 4, start:start + 4] = vector
        for start in [16, 18]:
            expected[start:start + 2, start:start + 2] = [[0, -2], [-1 / 2, 0]]
        error = np.max(np.abs(matrix - expected))
        assert error < 2e-12, (case['alpha'], case['chi'],
                               case['r'], case['oblique'], error)
        basis, speeds = canceled_basis(case['alpha'], case['W'])
        eigen_error = np.max(
            np.abs(np.einsum('ij,jk->ik', basis, matrix) - speeds[:, None] * basis))
        assert eigen_error < 1e-11, eigen_error
        assert np.linalg.matrix_rank(basis) == 20
        np.testing.assert_allclose(np.linalg.det(basis[:8, :8]),
                                   -24 * np.sqrt(case['q'] / case['f']), rtol=2e-13)
        maximum_error = max(maximum_error, error)
        maximum_eigen_error = max(maximum_eigen_error, eigen_error)
        maximum_condition = max(maximum_condition, np.linalg.cond(basis))
    # Exact old endpoint: multiplicity four, geometric multiplicity three.
    old = scalar_matrix(1, 3 / 4, 0, 0)
    for sign in [-1, 1]:
        assert 8 - np.linalg.matrix_rank(old - sign * np.eye(8)) == 3
    # Canceled family arbitrarily near and exactly at the harmonic endpoint.
    for alpha in [1e-3, 0.2, 1, 3, 10]:
        for w in [0, .3, .8, 1 - 1e-8, 1 - 1e-12, 1 - 1e-15, 1]:
            basis, speeds = canceled_basis(alpha, w)
            f, mu = 1 + (1 - w) * 2 / alpha, (1 - w) * 3 / 8 + w
            error = (np.einsum(
                'ij,jk->ik', basis[:8, :8], scalar_matrix(f, mu, w, w / 2))
                     - speeds[:8, None] * basis[:8, :8])
            assert np.max(np.abs(error)) < 1e-10
            assert np.isfinite(basis).all()
    print(json.dumps({'passed_kernel_cases': len(data),
                      'max_kernel_symbol_error': maximum_error,
                      'max_left_eigenfield_error': maximum_eigen_error,
                      'max_normalized_basis_condition': maximum_condition,
                      'old_endpoint_geometric_multiplicity': 3,
                      'harmonic_endpoint_complete': True}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
