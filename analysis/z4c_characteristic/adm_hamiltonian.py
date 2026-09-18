"""Independent ADM Hamiltonian diagnostic from physical metric derivatives.

Unlike a Hamiltonian-like quantity inferred from the Z4c Theta RHS, this
calculation does not use the independently evolved conformal connection.
It remains the ADM diagnostic when that connection violates its constraint.
Arrays may have arbitrary leading batch dimensions and complex dtype for
directional derivative checks. All tensor indices are covariant except the
explicit inverse metric and Christoffel symbols constructed below.
"""

import numpy as np


def ricci_tensor(metric, first, second):
    """Return the covariant Ricci tensor from physical metric derivatives.

    Trailing shapes are (3,3), (3,3,3), and (3,3,3,3), respectively.
    Derivative indices precede metric indices: first[...,a,i,j] = d_a g_ij.
    Inputs are read only. No independently evolved connection is used.
    """
    dtype = np.result_type(metric, first, second)
    inverse = np.linalg.inv(metric)
    lower_connection = np.empty(metric.shape[:-2] + (3, 3, 3), dtype=dtype)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                lower_connection[..., k, i, j] = 0.5 * (
                    first[..., i, k, j] + first[..., j, k, i] - first[..., k, i, j]
                )
    connection = np.einsum("...kl,...lij->...kij", inverse, lower_connection)
    derivative = -np.einsum("...kp,...apq,...qij->...akij", inverse, first, connection)
    for a in range(3):
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        derivative[..., a, k, i, j] += 0.5 * inverse[..., k, l] * (
                            second[..., a, i, l, j]
                            + second[..., a, j, l, i]
                            - second[..., a, l, i, j]
                        )
    ricci = np.zeros_like(metric, dtype=dtype)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ricci[..., i, j] += (
                    derivative[..., k, k, i, j] - derivative[..., j, k, i, k]
                )
                for l in range(3):
                    ricci[..., i, j] += (
                        connection[..., k, k, l] * connection[..., l, i, j]
                        - connection[..., k, j, l] * connection[..., l, i, k]
                    )
    return ricci


def hamiltonian(metric, first, second, extrinsic):
    """Return R + K**2 - K_ij K**ij in vacuum.

    Trailing shapes are (3,3), (3,3,3), (3,3,3,3), and (3,3), respectively.
    Derivative indices precede metric indices: first[...,a,i,j] = d_a g_ij.
    Inputs are read only. No constraint projection or background subtraction
    is performed here; callers must distinguish raw H from a residual of H.
    """
    ricci = ricci_tensor(metric, first, second)
    inverse = np.linalg.inv(metric)
    scalar_curvature = np.einsum("...ij,...ij->...", inverse, ricci)
    mixed_extrinsic = inverse @ extrinsic
    trace = np.trace(mixed_extrinsic, axis1=-2, axis2=-1)
    return scalar_curvature + trace**2 - np.einsum(
        "...ij,...ji->...", mixed_extrinsic, mixed_extrinsic
    )
