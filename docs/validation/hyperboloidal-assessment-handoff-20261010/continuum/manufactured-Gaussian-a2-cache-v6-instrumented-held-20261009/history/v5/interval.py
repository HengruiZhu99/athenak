"""Outward dyadic rational intervals; source-only until separately released."""
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt


def pow2(exponent):
    if type(exponent) is not int:
        raise TypeError("integer exponent required")
    return Fraction(1 << exponent, 1) if exponent >= 0 else Fraction(1, 1 << -exponent)


def floorlog2(value):
    value = Fraction(value)
    if value <= 0:
        raise ValueError("positive rational required")
    n, d = value.numerator, value.denominator
    e = n.bit_length() - d.bit_length()
    if (n < (d << e)) if e >= 0 else ((n << -e) < d):
        e -= 1
    return e


def directed(value, bits, upper):
    """Exact directed rounding to at most bits significant binary digits."""
    value = Fraction(value)
    if value == 0:
        return value
    if type(bits) is not int or bits < 8:
        raise ValueError("invalid precision")
    step = pow2(floorlog2(abs(value)) - bits + 1)
    x = value / step
    q, r = divmod(x.numerator, x.denominator)
    if upper and r:
        q += 1
    return q * step


def encode(value):
    """Exact dyadic [signed hexadecimal integer, exponent], no float."""
    value = Fraction(value)
    d = value.denominator
    if d & (d - 1):
        raise ValueError("non-dyadic certificate coefficient")
    n, exponent = value.numerator, -(d.bit_length() - 1)
    if n == 0:
        return ["0x0", 0]
    trailing = (abs(n) & -abs(n)).bit_length() - 1
    n //= 1 << trailing
    exponent += trailing
    return [hex(n), exponent]


def decode(record):
    if (not isinstance(record, list) or len(record) != 2
            or not isinstance(record[0], str) or type(record[1]) is not int):
        raise ValueError("invalid dyadic record")
    if len(record[0]) > 80 or abs(record[1]) > 100000:
        raise ValueError("oversized dyadic record")
    return int(record[0], 16) * pow2(record[1])


@dataclass(frozen=True)
class Interval:
    ctx: object
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    def __add__(self, other):
        other = self.ctx.cast(other)
        return self.ctx.enclose(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return self.ctx.enclose(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-self.ctx.cast(other))

    def __rsub__(self, other):
        return self.ctx.cast(other) + (-self)

    def __mul__(self, other):
        other = self.ctx.cast(other)
        values = [self.lo * other.lo, self.lo * other.hi,
                  self.hi * other.lo, self.hi * other.hi]
        return self.ctx.enclose(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.ctx.cast(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("zero-containing denominator")
        inverse = self.ctx.enclose(Fraction(1) / other.hi, Fraction(1) / other.lo)
        return self * inverse

    def __rtruediv__(self, other):
        return self.ctx.cast(other) / self

    def square(self):
        low = 0 if self.lo <= 0 <= self.hi else min(self.lo * self.lo, self.hi * self.hi)
        return self.ctx.enclose(low, max(self.lo * self.lo, self.hi * self.hi))

    def power(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("nonnegative integral power required")
        result, base = self.ctx.point(1), self
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base.square()
        return result


class Context:
    def __init__(self, bits, exp_endpoint_cap=4096):
        if type(bits) is not int or bits < 8:
            raise ValueError("invalid precision")
        if type(exp_endpoint_cap) is not int or not 0 <= exp_endpoint_cap <= 4096:
            raise ValueError("invalid exact endpoint cache cap")
        self.bits = bits
        self._exp_endpoint_cap = exp_endpoint_cap
        self._exp_endpoint_cache = {}
        self._exp_endpoint_audit = {"hits": 0, "misses": 0, "evictions": 0}

    def enclose(self, lo, hi=None):
        lo, hi = Fraction(lo), Fraction(lo if hi is None else hi)
        if lo > hi:
            raise ValueError("reversed interval")
        return Interval(self, directed(lo, self.bits, False), directed(hi, self.bits, True))

    def point(self, value):
        return self.enclose(value)

    def cast(self, value):
        if isinstance(value, Interval):
            if value.ctx is not self:
                raise ValueError("mixed interval contexts")
            return value
        return self.point(value)

    def sqrt(self, value):
        value = self.cast(value)
        if value.lo < 0:
            raise ValueError("negative sqrt interval")

        def endpoint(x, up):
            if x == 0:
                return Fraction(0)
            step = pow2(floorlog2(x) // 2 - self.bits + 1)
            a = x / (step * step)
            k = isqrt(a.numerator // a.denominator)
            exact = k * k * a.denominator == a.numerator
            return (k + int(up and not exact)) * step

        return self.enclose(endpoint(value.lo, False), endpoint(value.hi, True))

    def exp_neg(self, value):
        """Enclose exp(-value) by exact odd65/even64 alternating sums."""
        value = self.cast(value)
        if value.lo < 0:
            raise ValueError("negative argument for exp_neg")

        def endpoint(y):
            m = 0
            while y > Fraction(1, 16) * (1 << m):
                m += 1
                if m > 64:
                    raise ValueError("declared exp range-reduction bound exceeded")
            z = y / (1 << m)
            total, term, even, odd = Fraction(1), Fraction(1), None, None
            for n in range(1, 66):
                term = -term * z / n
                total += term
                if n == 64:
                    even = total
                if n == 65:
                    odd = total
            result = self.enclose(odd, even)
            if result.lo < 0:
                raise ValueError("nonpositive reduced exponential enclosure")
            for _ in range(m):
                result = result.square()
            return result

        def cached_endpoint(y):
            # Exact Fraction and precision keys, private to this Context.
            if not isinstance(y, Fraction):
                raise TypeError("exact Fraction endpoint required")
            key = (self.bits, y)
            if key in self._exp_endpoint_cache:
                self._exp_endpoint_audit["hits"] += 1
                result = self._exp_endpoint_cache[key]
                if result.ctx is not self:
                    raise RuntimeError("cross-context cached interval")
                return result
            self._exp_endpoint_audit["misses"] += 1
            result = endpoint(y)  # The original uncached body above is unchanged.
            if self._exp_endpoint_cap:
                if len(self._exp_endpoint_cache) >= self._exp_endpoint_cap:
                    oldest = next(iter(self._exp_endpoint_cache))
                    del self._exp_endpoint_cache[oldest]
                    self._exp_endpoint_audit["evictions"] += 1
                self._exp_endpoint_cache[key] = result
            return result

        lower, upper = cached_endpoint(value.hi), cached_endpoint(value.lo)
        return self.enclose(lower.lo, upper.hi)
