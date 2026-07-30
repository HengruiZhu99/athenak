#!/usr/bin/env python3
"""Dependency-light numerical checks of the residual Z4c characteristic algebra."""

import math
import random

import numpy as np


TOLERANCE = 1.0e-12


def scalar_matrix(n, c, lapse, shift):
    matrix = np.zeros((8, 8))
    matrix[0, 6] = -c
    matrix[1, 3] = n * c / 2.0
    matrix[1, 4] = n
    matrix[2, 3] = 2.0 * n * c / 3.0
    matrix[2, 4] = n / 3.0
    matrix[2, 5] = -n * c / 2.0
    matrix[2, 6] = -2.0 * c / 3.0
    matrix[3, 0] = -4.0 * n / 3.0
    matrix[3, 1] = -2.0 * n / 3.0
    matrix[3, 7] = 4.0 / 3.0
    matrix[4, 0] = 2.0 * c * n / 3.0
    matrix[4, 1] = 4.0 * c * n / 3.0
    matrix[4, 7] = -2.0 * c / 3.0
    matrix[5, 2] = -2.0 * n
    matrix[5, 7] = 4.0 / 3.0
    matrix[6, 0] = -lapse
    matrix[7, 3] = shift
    return matrix


def scalar_left(n, c, lapse, shift, sign):
    root_c = math.sqrt(c)
    root_lapse = math.sqrt(lapse)
    root_shift = math.sqrt(shift)
    d_lapse_shift = 3.0 * c * lapse - 4.0 * shift
    d_light_shift = 3.0 * c * n * n - 4.0 * shift
    p = np.zeros((4, 4))
    d = np.zeros((4, 4))
    p[0, 0] = -sign * root_lapse / root_c
    d[0, 2] = 1.0
    p[1, 0] = 4.0 * shift * n * d_light_shift
    p[1, 1] = 2.0 * shift * n * d_lapse_shift
    p[1, 3] = (
        sign
        * 2.0
        * math.sqrt(3.0)
        * root_shift
        * (c * n * n - shift)
        * d_lapse_shift
    )
    d[1, 0] = (
        sign * math.sqrt(3.0) * root_shift * n * n * d_lapse_shift
    )
    d[1, 2] = (
        -sign
        * 2.0
        * math.sqrt(3.0)
        * c
        * root_shift
        * n
        * d_light_shift
    )
    d[1, 3] = d_lapse_shift * d_light_shift
    p[2, 1] = sign * root_c
    p[2, 3] = c / 2.0
    d[2, 0] = 1.0
    p[3, 0] = sign * 4.0 / (3.0 * root_c)
    p[3, 1] = sign * 2.0 / (3.0 * root_c)
    p[3, 2] = -sign * 2.0 / root_c
    p[3, 3] = -1.0
    d[3, 1] = 1.0
    speeds = np.array(
        [
            sign * math.sqrt(c * lapse),
            sign * math.sqrt(4.0 * shift / 3.0),
            sign * n * root_c,
            sign * n * root_c,
        ]
    )
    return np.concatenate((p, d), axis=1), speeds


def vector_matrix(n, c, shift):
    return np.array(
        [
            [0.0, n * c / 2.0, -n * c / 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-2.0 * n, 0.0, 0.0, 1.0],
            [0.0, shift, 0.0, 0.0],
        ]
    )


def tensor_matrix(n, c):
    return np.array([[0.0, -n * c / 2.0], [-2.0 * n, 0.0]])


def tensor_left(c, sign):
    return np.array([[-sign * 2.0 / math.sqrt(c), 1.0]])


def hybrid_roots(full_beta, background_beta, q):
    discriminant = (full_beta - background_beta) ** 2 + 4.0 * q
    midpoint = 0.5 * (full_beta + background_beta)
    half_width = 0.5 * math.sqrt(discriminant)
    return midpoint + half_width, midpoint - half_width


def finite_scalar_left(
    n, c, lapse, shift, background_beta, full_beta, sign
):
    root_c = math.sqrt(c)
    lapse_root = hybrid_roots(
        full_beta, background_beta, c * lapse)[0 if sign > 0 else 1]
    shift_root = hybrid_roots(
        full_beta, background_beta, 4.0 * shift / 3.0
    )[0 if sign > 0 else 1]
    shift_mu = shift_root - full_beta
    shift_delta_bg = shift_root - background_beta
    lapse_separation = c * lapse - shift_mu * shift_delta_bg
    light_separation = c * n * n - shift_mu * shift_mu

    rows = np.zeros((4, 8))
    rows[0, 0] = -(lapse_root - background_beta) / c
    rows[0, 6] = 1.0
    rows[1, 0] = n * shift_delta_bg**2 / lapse_separation
    rows[1, 1] = (
        n * shift_mu * shift_delta_bg / (2.0 * light_separation)
    )
    rows[1, 3] = (
        shift_delta_bg * (4.0 * c * n * n - 3.0 * shift_mu**2)
        / (4.0 * light_separation)
    )
    rows[1, 4] = n * n * shift_delta_bg / (2.0 * light_separation)
    rows[1, 6] = -c * n * shift_delta_bg / lapse_separation
    rows[1, 7] = 1.0
    rows[1] *= lapse_separation * light_separation
    rows[2, 1] = sign * root_c
    rows[2, 3] = c / 2.0
    rows[2, 4] = 1.0
    rows[3, 0] = sign * 4.0 / (3.0 * root_c)
    rows[3, 1] = sign * 2.0 / (3.0 * root_c)
    rows[3, 2] = -sign * 2.0 / root_c
    rows[3, 3] = -1.0
    rows[3, 5] = 1.0
    speeds = np.array(
        [
            lapse_root,
            shift_root,
            full_beta + sign * n * root_c,
            full_beta + sign * n * root_c,
        ]
    )
    return rows, speeds


def check_finite_residual_algebra(
    label, n, c, lapse, shift, background_beta, full_beta
):
    """Check the exact hybrid-advection eigensystem and projectors."""
    scalar = scalar_matrix(n, c, lapse, shift)
    scalar += full_beta * np.eye(8)
    scalar[6, 6] = background_beta
    scalar[7, 7] = background_beta

    vector = vector_matrix(n, c, shift)
    vector += full_beta * np.eye(4)
    vector[3, 3] = background_beta

    tensor = tensor_matrix(n, c) + full_beta * np.eye(2)
    midpoint = 0.5 * (full_beta + background_beta)
    difference = full_beta - background_beta
    light = n * math.sqrt(c)
    expected_scalar = [
        full_beta - light,
        full_beta - light,
        full_beta + light,
        full_beta + light,
        midpoint - 0.5 * math.sqrt(difference**2 + 4.0 * c * lapse),
        midpoint + 0.5 * math.sqrt(difference**2 + 4.0 * c * lapse),
        midpoint - 0.5 * math.sqrt(
            difference**2 + 16.0 * shift / 3.0),
        midpoint + 0.5 * math.sqrt(
            difference**2 + 16.0 * shift / 3.0),
    ]
    expected_vector = [
        full_beta - light,
        full_beta + light,
        midpoint - 0.5 * math.sqrt(difference**2 + 4.0 * shift),
        midpoint + 0.5 * math.sqrt(difference**2 + 4.0 * shift),
    ]
    expected_tensor = [full_beta - light, full_beta + light]

    maximum_error = 0.0
    for block_name, matrix, expected in (
        ("scalar", scalar, expected_scalar),
        ("vector", vector, expected_vector),
        ("tensor", tensor, expected_tensor),
    ):
        numerical = np.linalg.eigvals(matrix)
        imaginary_error = float(np.max(np.abs(numerical.imag)))
        root_error = float(np.max(np.abs(
            np.sort(numerical.real) - np.sort(np.asarray(expected)))))
        error = max(imaginary_error, root_error)
        if not math.isfinite(error) or error > TOLERANCE:
            raise RuntimeError(
                f"{label}:finite-{block_name} eigenvalue error {error:.8e}"
            )
        maximum_error = max(maximum_error, error)

    scalar_positive, scalar_positive_speeds = finite_scalar_left(
        n, c, lapse, shift, background_beta, full_beta, 1.0)
    scalar_negative, scalar_negative_speeds = finite_scalar_left(
        n, c, lapse, shift, background_beta, full_beta, -1.0)
    maximum_error = max(
        maximum_error,
        check_block(
            f"{label}:finite-scalar",
            scalar,
            scalar_positive,
            scalar_positive_speeds,
            scalar_negative,
            scalar_negative_speeds,
        ),
    )
    scalar_p_map = scalar_positive[:, :4]
    rhs = np.array((0.125, -0.375, 0.625, -0.875))
    correction = np.zeros(4)
    correction[0] = rhs[0] / scalar_p_map[0, 0]
    gamma_denominator = (
        scalar_p_map[1, 3] -
        0.5 * math.sqrt(c) * scalar_p_map[1, 1]
    )
    correction[3] = (
        rhs[1] -
        scalar_p_map[1, 0] * correction[0] -
        scalar_p_map[1, 1] * rhs[2] / math.sqrt(c)
    ) / gamma_denominator
    correction[1] = (
        rhs[2] - 0.5 * c * correction[3]
    ) / math.sqrt(c)
    correction[2] = (
        rhs[3] -
        scalar_p_map[3, 0] * correction[0] -
        scalar_p_map[3, 1] * correction[1] -
        scalar_p_map[3, 3] * correction[3]
    ) / scalar_p_map[3, 2]
    sparse_solve_error = float(
        np.max(np.abs(scalar_p_map @ correction - rhs))
    )
    if sparse_solve_error > TOLERANCE:
        raise RuntimeError(
            f"{label}: finite sparse scalar solve error "
            f"{sparse_solve_error:.8e}")
    maximum_error = max(maximum_error, sparse_solve_error)

    vector_positive = np.array(
        [
            [
                0.0,
                hybrid_roots(
                    full_beta, background_beta, shift)[0] -
                background_beta,
                0.0,
                1.0,
            ],
            [-2.0 / math.sqrt(c), -1.0, 1.0, 0.0],
        ]
    )
    vector_negative = np.array(
        [
            [
                0.0,
                hybrid_roots(
                    full_beta, background_beta, shift)[1] -
                background_beta,
                0.0,
                1.0,
            ],
            [2.0 / math.sqrt(c), -1.0, 1.0, 0.0],
        ]
    )
    vector_positive_speeds = np.array(
        [
            hybrid_roots(full_beta, background_beta, shift)[0],
            full_beta + n * math.sqrt(c),
        ]
    )
    vector_negative_speeds = np.array(
        [
            hybrid_roots(full_beta, background_beta, shift)[1],
            full_beta - n * math.sqrt(c),
        ]
    )
    maximum_error = max(
        maximum_error,
        check_block(
            f"{label}:finite-vector",
            vector,
            vector_positive,
            vector_positive_speeds,
            vector_negative,
            vector_negative_speeds,
        ),
    )

    maximum_error = max(
        maximum_error,
        check_block(
            f"{label}:finite-tensor",
            tensor,
            tensor_left(c, 1.0),
            np.array([full_beta + n * math.sqrt(c)]),
            tensor_left(c, -1.0),
            np.array([full_beta - n * math.sqrt(c)]),
        ),
    )
    all_speeds = np.concatenate(
        (
            scalar_positive_speeds,
            scalar_negative_speeds,
            vector_positive_speeds,
            vector_negative_speeds,
        )
    )
    if not (
            np.all(scalar_positive_speeds > 0.0) and
            np.all(scalar_negative_speeds < 0.0) and
            np.all(vector_positive_speeds > 0.0) and
            np.all(vector_negative_speeds < 0.0) and
            np.all(np.isfinite(all_speeds))):
        raise RuntimeError(
            f"{label}: finite residual does not have one incoming root per pair")
    return maximum_error


def check_block(
    name,
    matrix,
    positive_left,
    positive_speeds,
    negative_left,
    negative_speeds,
):
    left = np.concatenate((positive_left, negative_left), axis=0)
    speeds = np.concatenate((positive_speeds, negative_speeds))
    right = np.linalg.inv(left)
    identity = np.eye(left.shape[0])
    errors = [
        np.max(np.abs(left @ right - identity)),
        np.max(np.abs(left @ matrix - speeds[:, None] * left)),
    ]
    incoming_count = positive_left.shape[0]
    projector = right[:, :incoming_count] @ left[:incoming_count, :]
    errors.append(np.max(np.abs(projector @ projector - projector)))
    errors.append(np.max(np.abs(negative_left @ projector)))
    rng = np.random.RandomState(314159)
    state = rng.normal(size=left.shape[1])
    errors.append(
        np.max(
            np.abs(
                negative_left @ ((identity - projector) @ state)
                - negative_left @ state
            )
        )
    )
    error = float(max(errors))
    if not math.isfinite(error) or error > TOLERANCE:
        raise RuntimeError(f"{name} algebra error {error:.8e}")
    return error


def check_metric_frame(metric, side):
    inverse = np.linalg.inv(metric)
    side = np.asarray(side, dtype=float)
    if not np.any(side):
        side[0] = 1.0
    normal_d = side / math.sqrt(float(side @ inverse @ side))
    normal_u = inverse @ normal_d
    # Store the three projected coordinate basis vectors by row.  The
    # mixed-index projector P^i_j has projected e_(j) in its columns.
    candidates = (np.eye(3) - np.outer(normal_u, normal_d)).T
    norms = np.einsum("ai,ij,aj->a", candidates, metric, candidates)
    tangent1_u = candidates[int(np.argmax(norms))]
    tangent1_u /= math.sqrt(float(tangent1_u @ metric @ tangent1_u))
    tangent1_d = metric @ tangent1_u
    tangent2_u = np.cross(normal_d, tangent1_d) / math.sqrt(
        float(np.linalg.det(metric))
    )
    tangent2_u /= math.sqrt(float(tangent2_u @ metric @ tangent2_u))
    frame = np.stack((normal_u, tangent1_u, tangent2_u))
    return float(np.max(np.abs(frame @ metric @ frame.T - np.eye(3))))


def check_random_frame(rng):
    raw = rng.normal(size=(3, 3))
    metric = raw.T @ raw + 0.25 * np.eye(3)
    side = rng.choice((-1.0, 0.0, 1.0), size=3)
    return check_metric_frame(metric, side)


def kerr_schild_case(label, mass, spin, xyz, side):
    """Return actual Kerr-Schild conformal coefficients at one Cartesian point."""
    x, y, z = xyz
    rho2 = x * x + y * y + z * z
    discriminant = (rho2 - spin * spin) ** 2 + 4.0 * spin * spin * z * z
    r2 = 0.5 * (
        rho2 - spin * spin + math.sqrt(discriminant))
    r = math.sqrt(r2)
    denominator = r2 + spin * spin
    null_spatial = np.array(
        [
            (r * x + spin * y) / denominator,
            (r * y - spin * x) / denominator,
            z / r,
        ]
    )
    h = mass * r ** 3 / (r ** 4 + spin * spin * z * z)
    physical_metric = np.eye(3) + 2.0 * h * np.outer(
        null_spatial, null_spatial)
    determinant = float(np.linalg.det(physical_metric))
    expected_determinant = 1.0 + 2.0 * h
    if not math.isclose(
            determinant, expected_determinant,
            rel_tol=1.0e-13, abs_tol=1.0e-15):
        raise RuntimeError(
            "{}: Kerr-Schild determinant mismatch".format(label))
    chi = determinant ** (-1.0 / 3.0)
    conformal_metric = chi * physical_metric
    if not math.isclose(
            float(np.linalg.det(conformal_metric)), 1.0,
            rel_tol=1.0e-13, abs_tol=1.0e-15):
        raise RuntimeError(
            "{}: conformal metric does not have unit determinant".format(
                label))
    alpha = 1.0 / math.sqrt(expected_determinant)
    beta = (
        2.0 * h / expected_determinant) * null_spatial
    inverse_conformal = np.linalg.inv(conformal_metric)
    side_covector = np.asarray(side, dtype=float)
    normal_covector = side_covector / math.sqrt(
        float(side_covector @ inverse_conformal @ side_covector))
    beta_normal = float(normal_covector @ beta)
    lapse_driver = 2.0 * alpha
    return (
        (label, alpha, chi, lapse_driver, 1.0, beta_normal),
        conformal_metric,
        side,
    )


def check_case(label, n, c, lapse, shift, beta):
    scalar_positive, scalar_positive_speeds = scalar_left(
        n, c, lapse, shift, 1.0
    )
    scalar_negative, scalar_negative_speeds = scalar_left(
        n, c, lapse, shift, -1.0
    )
    errors = [
        check_block(
            f"{label}:scalar",
            scalar_matrix(n, c, lapse, shift),
            scalar_positive,
            scalar_positive_speeds,
            scalar_negative,
            scalar_negative_speeds,
        )
    ]

    vector_positive = np.array(
        [
            [0.0, math.sqrt(shift), 0.0, 1.0],
            [-2.0 / math.sqrt(c), -1.0, 1.0, 0.0],
        ]
    )
    vector_negative = np.array(
        [
            [0.0, -math.sqrt(shift), 0.0, 1.0],
            [2.0 / math.sqrt(c), -1.0, 1.0, 0.0],
        ]
    )
    vector_speeds = np.array([math.sqrt(shift), n * math.sqrt(c)])
    errors.append(
        check_block(
            f"{label}:vector",
            vector_matrix(n, c, shift),
            vector_positive,
            vector_speeds,
            vector_negative,
            -vector_speeds,
        )
    )

    tensor_positive = tensor_left(c, 1.0)
    tensor_negative = tensor_left(c, -1.0)
    tensor_speed = np.array([n * math.sqrt(c)])
    errors.append(
        check_block(
            f"{label}:tensor",
            tensor_matrix(n, c),
            tensor_positive,
            tensor_speed,
            tensor_negative,
            -tensor_speed,
        )
    )
    minimum_speed = min(
        math.sqrt(c * lapse),
        math.sqrt(4.0 * shift / 3.0),
        n * math.sqrt(c),
        math.sqrt(shift),
    )
    if abs(beta) >= minimum_speed:
        raise RuntimeError(f"{label} does not have one incoming member per pair")
    # Exercise both signs of the finite residual shift, rather than testing
    # only full_beta > background_beta.
    shift_sign = -1.0 if sum(ord(character) for character in label) % 2 else 1.0
    full_beta = beta + shift_sign * 0.125 * minimum_speed
    errors.append(
        check_finite_residual_algebra(
            label, n, c, lapse, shift, beta, full_beta
        )
    )
    return max(errors)


cases = [
    ("minkowski_muS1", 1.0, 1.0, 2.0, 1.0, 0.0),
]
ks_cases = [
    kerr_schild_case(
        "schwarzschild_x_face", 1.0, 0.0, (12.8, 0.0, 0.0),
        (1.0, 0.0, 0.0)),
    kerr_schild_case(
        "kerr_a09_y_face", 1.0, 0.9, (20.0, 7.0, 5.0),
        (0.0, 1.0, 0.0)),
    kerr_schild_case(
        "kerr_a09_corner", 1.0, 0.9, (-8.0, 11.0, 4.0),
        (-1.0, 1.0, 1.0)),
]
cases.extend(case for case, _, _ in ks_cases)

python_rng = random.Random(4815162342)
while len(cases) < 103:
    n = python_rng.uniform(0.4, 1.2)
    c = python_rng.uniform(0.5, 1.5)
    lapse = python_rng.uniform(0.5, 2.5)
    shift = python_rng.uniform(0.4, 1.4)
    d_lapse = abs(3.0 * c * lapse - 4.0 * shift)
    d_light = abs(3.0 * c * n * n - 4.0 * shift)
    if min(d_lapse, d_light) < 0.1:
        continue
    minimum_speed = min(
        math.sqrt(c * lapse),
        math.sqrt(4.0 * shift / 3.0),
        n * math.sqrt(c),
        math.sqrt(shift),
    )
    beta = python_rng.uniform(-0.8 * minimum_speed, 0.8 * minimum_speed)
    cases.append((f"random_{len(cases):03d}", n, c, lapse, shift, beta))

maximum_error = 0.0
for case in cases:
    maximum_error = max(maximum_error, check_case(*case))

numpy_rng = np.random.RandomState(271828)
maximum_frame_error = max(check_random_frame(numpy_rng) for _ in range(100))
maximum_frame_error = max(
    maximum_frame_error,
    max(check_metric_frame(metric, side) for _, metric, side in ks_cases),
)
if maximum_frame_error > TOLERANCE:
    raise RuntimeError(f"boundary-frame error {maximum_frame_error:.8e}")

print(
    f"PASS cases={len(cases)} algebra_error={maximum_error:.8e} "
    f"frame_error={maximum_frame_error:.8e}"
)
