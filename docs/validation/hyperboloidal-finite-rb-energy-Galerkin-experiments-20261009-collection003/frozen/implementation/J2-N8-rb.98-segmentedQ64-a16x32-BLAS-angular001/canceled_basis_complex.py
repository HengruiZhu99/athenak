# Exact frozen scalar/canceled-basis formulas; only zero-array dtype generalized for complex-step differentiation.
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
    basis = np.zeros((20, 20), dtype=np.result_type(alpha, w, float))
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


