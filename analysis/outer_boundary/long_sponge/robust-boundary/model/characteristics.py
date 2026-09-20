"""Selected characteristic functions vendored verbatim from the immutable
production snapshot's analysis/z4c_characteristic/
check_residual_characteristics_numeric.py (8b694211 + static-floor.patch).
Original project license: BSD 3-Clause; see the repository LICENSE.
Only packaging changed; function bodies are verbatim.
Source SHA256: d3d4318b5c5575c43f3320ab029cf3534cc3b3e1e8f3859e0df872aa454a5a79
"""
import math
import numpy as np

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
