"""Vacuum ADM momentum covector from physical metric and extrinsic curvature.

This diagnostic never uses the independently evolved conformal connection.
Leading batch dimensions and complex directional derivatives are supported.
"""

import numpy as np


def momentum(metric, first, extrinsic, first_extrinsic):
    """Return D_j K^j_i - D_i K, without projection or subtraction.

    All inputs have covariant tensor indices. Derivative indices come first:
    first[...,a,i,j] = partial_a metric_ij, and likewise first_extrinsic.
    Trailing shapes are (3,3), (3,3,3), (3,3), and (3,3,3).
    Inputs are read only; matter momentum must be subtracted by the caller.
    """
    inverse = np.linalg.inv(metric)
    dtype = np.result_type(metric, first, extrinsic, first_extrinsic)
    lower = np.empty(metric.shape[:-2] + (3, 3, 3), dtype=dtype)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                lower[..., k, i, j] = 0.5 * (
                    first[..., i, k, j] + first[..., j, k, i] - first[..., k, i, j]
                )
    connection = np.einsum("...kl,...lij->...kij", inverse, lower)
    derivative = first_extrinsic.copy().astype(dtype)
    derivative -= np.einsum("...lai,...lj->...aij", connection, extrinsic)
    derivative -= np.einsum("...laj,...il->...aij", connection, extrinsic)
    return (np.einsum("...jk,...jki->...i", inverse, derivative)
            - np.einsum("...jk,...ijk->...i", inverse, derivative))
