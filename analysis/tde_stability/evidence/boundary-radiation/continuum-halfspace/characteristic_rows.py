"""Frozen scalar incoming rows copied from the repository characteristic audit.

Parameters: lapse alpha, conformal chi, lapse-driver coefficient, shift driver,
and incoming (+) or outgoing (-) normal branch. This local copy makes the
half-space evidence scripts independent of the historical review directory.
"""
import math
import numpy as np

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
