"""Small complete spacetime Taylor algebra with explicit derivative order."""
import itertools
import math

import mpmath as mp

ZERO = (0, 0, 0, 0)


def indices(order):
    return [m for m in itertools.product(range(order+1), repeat=4) if sum(m) <= order]


def unit(axis):
    return tuple(int(i == axis) for i in range(4))


class Jet:
    def __init__(self, value, order=2, coefficients=None):
        self.order = order
        self.c = {m: mp.mpf(0) for m in indices(order)}
        self.c[ZERO] = mp.mpf(value)
        if coefficients is not None:
            for m, x in coefficients.items():
                assert sum(m) <= order
                self.c[m] = mp.mpf(x)

    @property
    def v(self):
        return self.c[ZERO]

    def derivative(self, *axes):
        m = tuple(axes.count(i) for i in range(4))
        assert sum(m) <= self.order
        return self.c[m]*math.prod(math.factorial(k) for k in m)

    def diff(self, axis):
        assert self.order > 0, 'No absent derivative may be padded'
        out = Jet(0, self.order-1)
        for m in out.c:
            plus = list(m)
            plus[axis] += 1
            out.c[m] = self.c[tuple(plus)]*(m[axis]+1)
        return out

    def truncate(self, order):
        assert order <= self.order
        return Jet(self.v, order, {m: x for m, x in self.c.items() if sum(m) <= order})

    def coerce(self, other):
        return other if isinstance(other, Jet) else Jet(other, self.order)

    def __add__(self, other):
        other = self.coerce(other)
        n = min(self.order, other.order)
        return Jet(0, n, {m: self.c[m]+other.c[m] for m in indices(n)})

    __radd__ = __add__

    def __neg__(self):
        return Jet(0, self.order, {m: -x for m, x in self.c.items()})

    def __sub__(self, other):
        return self+-self.coerce(other)

    def __rsub__(self, other):
        return -self+other

    def __mul__(self, other):
        other = self.coerce(other)
        n = min(self.order, other.order)
        out = Jet(0, n)
        for a, x in self.c.items():
            for b, y in other.c.items():
                m = tuple(i+j for i, j in zip(a, b))
                if sum(m) <= n:
                    out.c[m] += x*y
        return out

    __rmul__ = __mul__

    def __pow__(self, power):
        assert self.v != 0
        if power != int(power):
            assert self.v > 0
        u = self*(1/self.v)-1
        out = Jet(1, self.order)
        term = Jet(1, self.order)
        for k in range(1, self.order+1):
            term = term*u
            out += mp.binomial(power, k)*term
        return self.v**power*out

    def __truediv__(self, other):
        return self*self.coerce(other)**(-1)

    def __rtruediv__(self, other):
        return self.coerce(other)*self**(-1)


def from_ordinary(value, first, second):
    c = {ZERO: value}
    for i in range(4):
        c[unit(i)] = first[i]
        for j in range(i, 4):
            m = tuple(int(k == i)+int(k == j) for k in range(4))
            c[m] = second[i][j]/(2 if i == j else 1)
    return Jet(value, 2, c)


def det3(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def inverse3(a):
    determinant = det3(a)
    result = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rows = [k for k in range(3) if k != j]
            cols = [k for k in range(3) if k != i]
            minor = a[rows[0]][cols[0]]*a[rows[1]][cols[1]]-a[rows[0]][cols[1]]*a[rows[1]][cols[0]]
            result[i][j] = (-1)**(i+j)*minor/determinant
    return result


def export(jet, digits):
    return {'order': jet.order,
            'ordinary': [{'multiindex': list(m),
                          'value': mp.nstr(x*math.prod(math.factorial(k) for k in m), digits, strip_zeros=False)}
                         for m, x in sorted(jet.c.items())]}
