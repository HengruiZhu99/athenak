"""HELD complete four-variable Taylor algebra; import only after admission.

Coefficients contain factorial denominators. No unavailable derivative is
padded: differentiation reduces order, and truncation can only reduce order.
"""
from functools import lru_cache
import itertools
import math
import mpmath as mp

ZERO = (0, 0, 0, 0)


@lru_cache(maxsize=4)
def indices(order):
    if type(order) is not int or not 0 <= order <= 3:
        raise ValueError("only complete orders zero through three are supported")
    return tuple(m for m in itertools.product(range(order + 1), repeat=4) if sum(m) <= order)


def factorial(m):
    return math.prod(math.factorial(k) for k in m)


@lru_cache(maxsize=4)
def products(order):
    result = []
    for m in indices(order):
        pairs = []
        for a in itertools.product(*(range(k + 1) for k in m)):
            b = tuple(m[i] - a[i] for i in range(4))
            pairs.append((a, b))
        result.append((m, tuple(pairs)))
    return tuple(result)


class Jet:
    def __init__(self, value, order=3, coefficients=None):
        self.order = order
        self.c = {m: mp.mpf(0) for m in indices(order)}
        self.c[ZERO] = mp.mpf(value)
        if coefficients is not None:
            for m, coefficient in coefficients.items():
                if m not in self.c:
                    raise ValueError("coefficient exceeds the explicit derivative order")
                self.c[m] = mp.mpf(coefficient)

    @property
    def v(self):
        return self.c[ZERO]

    @classmethod
    def variable(cls, value, axis, order=3):
        if type(axis) is not int or not 0 <= axis < 4 or order == 0:
            raise ValueError("invalid independent variable")
        m = tuple(int(i == axis) for i in range(4))
        return cls(value, order, {m: 1})

    def derivative(self, *axes):
        if any(type(i) is not int or not 0 <= i < 4 for i in axes) or len(axes) > self.order:
            raise ValueError("requested derivative is unavailable")
        m = tuple(axes.count(i) for i in range(4))
        return self.c[m] * factorial(m)

    def diff(self, axis):
        if self.order == 0 or type(axis) is not int or not 0 <= axis < 4:
            raise ValueError("cannot differentiate an unavailable order/axis")
        result = Jet(0, self.order - 1)
        for m in result.c:
            shifted = tuple(m[i] + int(i == axis) for i in range(4))
            result.c[m] = self.c[shifted] * (m[axis] + 1)
        return result

    def truncate(self, order):
        if type(order) is not int or not 0 <= order <= self.order:
            raise ValueError("truncation may not invent derivatives")
        return Jet(0, order, {m: self.c[m] for m in indices(order)})

    def coerce(self, value):
        return value if isinstance(value, Jet) else Jet(value, self.order)

    def __add__(self, other):
        other = self.coerce(other)
        order = min(self.order, other.order)
        return Jet(0, order, {m: self.c[m] + other.c[m] for m in indices(order)})

    __radd__ = __add__

    def __neg__(self):
        return Jet(0, self.order, {m: -value for m, value in self.c.items()})

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        other = self.coerce(other)
        order = min(self.order, other.order)
        return Jet(0, order, {m: mp.fsum(self.c[a] * other.c[b] for a, b in pairs)
                              for m, pairs in products(order)})

    __rmul__ = __mul__

    def __pow__(self, power):
        p = mp.mpf(power)
        if p == int(p) and p >= 0:
            # Polynomial powers are defined at a zero base, including 0^0=1.
            result = Jet(1, self.order)
            factor, count = self, int(p)
            while count:
                if count % 2:
                    result = result * factor
                factor = factor * factor
                count //= 2
            return result
        if self.v == 0 or (p != int(p) and self.v <= 0):
            raise ArithmeticError("invalid real power at the base point")
        u = self * (1 / self.v) - 1
        result, term = Jet(1, self.order), Jet(1, self.order)
        for k in range(1, self.order + 1):
            term = term * u
            result = result + mp.binomial(p, k) * term
        return self.v ** p * result

    def __truediv__(self, other):
        other = self.coerce(other)
        if other.v == 0:
            raise ArithmeticError("zero denominator")
        return self * other ** (-1)

    def __rtruediv__(self, other):
        return self.coerce(other) * self ** (-1)

    def exp(self):
        u = self - self.v
        result, term = Jet(1, self.order), Jet(1, self.order)
        for k in range(1, self.order + 1):
            term = term * u
            result = result + term / math.factorial(k)
        return mp.exp(self.v) * result


def compose_radial(value, first, second, third, radius):
    if radius.order != 3:
        raise ValueError("complete radius order three is required")
    d = radius - radius.v
    return value + first * d + second * d ** 2 / 2 + third * d ** 3 / 6


def compose_jet(jet, increments):
    """Substitute complete zero-base coordinate increments at equal order."""
    if len(increments) != 4 or any(d.order != jet.order or d.v != 0 for d in increments):
        raise ValueError('composition needs four equal-order, zero-base increments')
    result = Jet(0,jet.order)
    for m in indices(jet.order):
        term = Jet(jet.c[m],jet.order)
        for axis in range(4):
            term = term*increments[axis]**m[axis]
        result = result+term
    return result


def determinant(matrix):
    n = len(matrix)
    if n not in (3, 4) or any(len(row) != n for row in matrix):
        raise ValueError("only complete three/four square matrices")
    result = matrix[0][0] * 0
    for permutation in itertools.permutations(range(n)):
        inversions = sum(permutation[i] > permutation[j] for i in range(n) for j in range(i + 1, n))
        term = matrix[0][permutation[0]]
        for i in range(1, n):
            term = term * matrix[i][permutation[i]]
        result = result + (-1) ** inversions * term
    return result


def inverse(matrix):
    n = len(matrix)
    det = determinant(matrix)
    if det.v == 0:
        raise ArithmeticError("singular matrix")
    result = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rows, cols = [k for k in range(n) if k != j], [k for k in range(n) if k != i]
            if n == 3:
                minor = matrix[rows[0]][cols[0]] * matrix[rows[1]][cols[1]] - matrix[rows[0]][cols[1]] * matrix[rows[1]][cols[0]]
            else:
                minor = determinant([[matrix[a][b] for b in cols] for a in rows])
            result[i][j] = (-1) ** (i + j) * minor / det
    return result


def export(jet, digits):
    return {"order": jet.order,
            "ordinary": [{"multiindex": list(m), "value": mp.nstr(jet.c[m] * factorial(m), digits, strip_zeros=False)}
                         for m in indices(jet.order)]}
