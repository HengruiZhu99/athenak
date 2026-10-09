"""SOURCE-ONLY ordinary analytic jets; no finite differences or autodiff package.

This small algebra stores ordinary derivatives, not Taylor coefficients.
Only addition, multiplication and elementary analytic chain rules are used.
The independently displayed initial/CMC oracles do not use the ray quotient.
"""
import mpmath as mp


class Jet2:
    def __init__(self, value, gradient=None, hessian=None, dimension=3):
        self.v = mp.mpf(value)
        self.n = len(gradient) if gradient is not None else dimension
        self.d = list(gradient) if gradient is not None else [mp.mpf(0)]*self.n
        self.h = [list(row) for row in hessian] if hessian is not None else [[mp.mpf(0)]*self.n for _ in range(self.n)]

    @classmethod
    def variable(cls, value, index, dimension=3):
        d = [mp.mpf(int(i == index)) for i in range(dimension)]
        return cls(value, d)

    def lift(self, other):
        if isinstance(other, Jet2):
            if self.n != other.n:
                raise ValueError("jet dimension mismatch")
            return other
        return Jet2(other, dimension=self.n)

    def __add__(self, other):
        o = self.lift(other)
        return Jet2(self.v+o.v, [self.d[i]+o.d[i] for i in range(self.n)], [[self.h[i][j]+o.h[i][j] for j in range(self.n)] for i in range(self.n)])

    __radd__ = __add__

    def __neg__(self):
        return Jet2(-self.v, [-x for x in self.d], [[-x for x in row] for row in self.h])

    def __sub__(self, other):
        return self+-self.lift(other)

    def __rsub__(self, other):
        return self.lift(other)+-self

    def __mul__(self, other):
        o = self.lift(other)
        return Jet2(self.v*o.v, [self.d[i]*o.v+self.v*o.d[i] for i in range(self.n)], [[self.h[i][j]*o.v+self.d[i]*o.d[j]+self.d[j]*o.d[i]+self.v*o.h[i][j] for j in range(self.n)] for i in range(self.n)])

    __rmul__ = __mul__

    def compose(self, value, first, second):
        return Jet2(value, [first*x for x in self.d], [[first*self.h[i][j]+second*self.d[i]*self.d[j] for j in range(self.n)] for i in range(self.n)])

    def __pow__(self, power):
        p = mp.mpf(power)
        if p == 0:
            return Jet2(1, dimension=self.n)
        if p == 1:
            return self
        return self.compose(self.v**p, p*self.v**(p-1), p*(p-1)*self.v**(p-2))

    def __truediv__(self, other):
        return self*self.lift(other)**-1

    def __rtruediv__(self, other):
        return self.lift(other)*self**-1

    def exp(self):
        e = mp.exp(self.v)
        return self.compose(e, e, e)

    def flat(self):
        return [self.v]+self.d+[self.h[i][j] for i in range(self.n) for j in range(i, self.n)]


def radial_lift(value, first, second, q, nu):
    if not q > 0:
        raise ValueError("radial lift requires positive q; use the Cartesian core")
    return Jet2(value, [first*x for x in nu], [[second*nu[i]*nu[j]+first/q*(int(i == j)-nu[i]*nu[j]) for j in range(3)] for i in range(3)])


def sumjet(items, dimension=3):
    return sum(items, Jet2(0, dimension=dimension))
