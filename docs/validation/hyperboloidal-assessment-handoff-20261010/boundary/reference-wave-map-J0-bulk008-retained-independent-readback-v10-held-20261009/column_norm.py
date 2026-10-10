"""Finite positive Euclidean column norms without unscaled NumPy squaring."""
import math


def finite_positive_column2_norm(np, array):
    values = np.asarray(array)
    if values.ndim != 2 or min(values.shape) <= 0 or values.dtype != np.dtype('float64'):
        raise ValueError('column2-norm requires a nonempty float64 matrix')
    if not np.isfinite(values).all():
        raise ValueError('column2-norm input must be finite')
    norms = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        try:
            value = math.hypot(*(float(x) for x in values[:, column]))
        except OverflowError as error:
            raise ValueError('column2-norm overflow') from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError('column2-norm must be finite and positive')
        norms[column] = value
    return norms
