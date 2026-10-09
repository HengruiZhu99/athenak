"""Real binary64 fixed [13/13] Pade scaling-and-squaring candidate.

This independently written scratch helper uses the degree-13 evaluation in
Al-Mohy and Higham (2009), equations (3.5)--(3.6), and the norm-one scaling
threshold for Algorithm 3.1, Table 3.1.  It always uses degree 13 rather than
the algorithm's optional lower degrees.  It is NOT Algorithm 5.1 and does not
use its different theta_13=4.25 or adaptive power-norm estimator.

Every matrix product is a literal, unoptimized NumPy einsum.  The rational
approximant is evaluated with numpy.linalg.solve, without forming an inverse.
No warning is suppressed; nonfinite arithmetic or underflow is an error.
This implementation does not certify forward accuracy for arbitrary nonnormal
matrices.  In particular, norm-based overscaling and squaring roundoff remain.

Primary source: https://eprints.maths.manchester.ac.uk/1217/1/paper9.pdf
Original method: https://doi.org/10.1137/04061101X
"""

import math

import numpy as np


THETA13 = 5.371920351148152
COEFFICIENTS = (
    64764752532480000, 32382376266240000, 7771770303897600,
    1187353796428800, 129060195264000, 10559470521600,
    670442572800, 33522128640, 1323241920, 40840800,
    960960, 16380, 182, 1,
)


def mm(left, right):
    """Literal matrix product; deliberately no BLAS dispatch/optimized path."""
    return np.einsum("ik,kj->ij", left, right, optimize=False)


def one_norm(a):
    """Matrix one norm, with an explicit empty-matrix convention."""
    if a.shape[0] == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(a), axis=0)))


def _finite(a, label):
    if not np.all(np.isfinite(a)):
        raise FloatingPointError("nonfinite " + label)


def expm_pade13(argument, return_info=False):
    """Return exp(argument), optionally with scaling/solve diagnostics.

    The admitted input is a finite real square array convertible to float64.
    Norm overflow, intermediate overflow/underflow, singular solves, and
    nonfinite output raise exceptions.  There is no clipping, fallback,
    balancing, Schur transformation, eigenvalue computation, or warning filter.
    The reported residual is the normalized rational linear-solve residual;
    it is not a bound on matrix-exponential forward error.
    """
    raw = np.asarray(argument)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("argument must be a square rank-two array")
    if np.iscomplexobj(raw):
        raise TypeError("only real matrices are admitted")
    with np.errstate(all="raise"):
        a = np.array(raw, dtype=np.float64, copy=True)
        _finite(a, "argument")
        n = a.shape[0]
        identity = np.eye(n, dtype=np.float64)
        norm1 = one_norm(a)
        if norm1 == 0.0:
            info = {"degree": 13, "squarings": 0, "argument_one_norm": 0.0,
                    "scaled_one_norm": 0.0, "rational_solve_residual": 0.0,
                    "matrix_products": 0}
            return (identity, info) if return_info else identity
        s = max(0, int(math.ceil(math.log2(norm1 / THETA13))))
        a = np.ldexp(a, -s)
        # A rounded logarithm exactly at a power-of-two boundary can be low.
        # Enforce the published scaling inequality without changing theta_13.
        if one_norm(a) > THETA13:
            s += 1
            a = np.ldexp(a, -1)
        scaled_norm = one_norm(a)
        a2 = mm(a, a)
        a4 = mm(a2, a2)
        a6 = mm(a4, a2)
        b = COEFFICIENTS
        u_inner = (mm(a6, b[13] * a6 + b[11] * a4 + b[9] * a2)
                   + b[7] * a6 + b[5] * a4 + b[3] * a2 + b[1] * identity)
        u = mm(a, u_inner)
        v = (mm(a6, b[12] * a6 + b[10] * a4 + b[8] * a2)
             + b[6] * a6 + b[4] * a4 + b[2] * a2 + b[0] * identity)
        denominator, numerator = v - u, v + u
        _finite(denominator, "Pade denominator")
        _finite(numerator, "Pade numerator")
        result = np.linalg.solve(denominator, numerator)
        _finite(result, "rational approximant")
        residual = one_norm(mm(denominator, result) - numerator)
        residual /= one_norm(denominator) * one_norm(result) + one_norm(numerator)
        for _ in range(s):
            result = mm(result, result)
            _finite(result, "squaring result")
        info = {"degree": 13, "squarings": s, "argument_one_norm": norm1,
                "scaled_one_norm": scaled_norm,
                "rational_solve_residual": residual,
                "matrix_products": 7 + s}
        return (result, info) if return_info else result
