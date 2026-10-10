"""Scoped binary64 normalization arithmetic; importing this uses only stdlib.

Only possible tiny products/quotients use Fraction arithmetic. Ordinary
operations retain the supplied NumPy operation. No floating error is ignored.
"""
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import struct


MIN_NORMAL = Fraction(1, 1 << 1022)
MIN_SUBNORMAL = Fraction(1, 1 << 1074)


def fraction_record(value):
    value = Fraction(value)
    with localcontext() as ctx:
        ctx.prec = 45
        display = str(Decimal(value.numerator) / Decimal(value.denominator))
    return {'numerator': str(value.numerator),
            'denominator': str(value.denominator),
            'decimal_display_not_bound': display}


def rounded_binary64(value, negative_zero=False):
    """Round an exact rational to nearest binary64, ties to even.

    Integer rounding and bit construction avoid underflowing floating
    intermediates. Rounded overflow raises; finite-only callers remain strict.
    A nonzero exact value supplies its own sign, including when rounded to zero.
    """
    value = Fraction(value)
    negative = value < 0 or (value == 0 and negative_zero)
    n, d = abs(value.numerator), value.denominator
    if n == 0:
        bits = int(negative) << 63
        return struct.unpack('>d', struct.pack('>Q', bits))[0]
    exponent = n.bit_length() - d.bit_length()
    if exponent >= 0:
        if n < (d << exponent):
            exponent -= 1
    elif (n << (-exponent)) < d:
        exponent -= 1
    grid = max(exponent - 52, -1074)
    numerator, denominator = (n << (-grid), d) if grid < 0 else (n, d << grid)
    q, rem = divmod(numerator, denominator)
    if 2 * rem > denominator or (2 * rem == denominator and q % 2):
        q += 1
    if grid == -1074 and q < (1 << 52):
        bits = q
    else:
        # This also covers a subnormal rounding upward to the minimum normal.
        exponent = grid + 52
        if q == (1 << 53):
            q >>= 1
            exponent += 1
        if exponent > 1023:
            raise OverflowError('exact rational rounds beyond finite binary64')
        if not ((1 << 52) <= q < (1 << 53) and exponent >= -1022):
            raise ArithmeticError('invalid normal binary64 rounding state')
        bits = ((exponent + 1023) << 52) | (q - (1 << 52))
    bits |= int(negative) << 63
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def possible_tiny(a, b, operation):
    """Conservative exponent-only predicate; it never computes a*b or a/b.

    frexp gives .5<=|mantissa|<1. Products with ea+eb>=-1020 and
    quotients with ea-eb>=-1021 are necessarily normal. Zero operands
    do not underflow. Invalid operands/divisors are rejected explicitly.
    """
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError('finite binary64 normalization operands required')
    if operation not in ('multiply', 'divide'):
        raise ValueError('only product/quotient classification is supported')
    if operation == 'divide' and b == 0:
        raise ZeroDivisionError('normalization divisor is zero')
    if a == 0 or b == 0:
        return False
    ea, eb = math.frexp(a)[1], math.frexp(b)[1]
    return ea + eb <= -1021 if operation == 'multiply' else ea - eb <= -1022


class TinyArithmetic:
    """Exact fallback kernel plus an auditable per-operation rounding record."""
    def __init__(self):
        self.counts = Counter()
        self.labels = Counter()
        self.max_error = Fraction(0)
        self.max_error_case = None
        self.examples = []

    def exact(self, a, b, operation, label):
        a, b = float(a), float(b)
        if not possible_tiny(a, b, operation):
            raise ValueError('exact fallback is restricted to potential tiny results')
        aa, bb = Fraction.from_float(a), Fraction.from_float(b)
        value = aa * bb if operation == 'multiply' else aa / bb
        result = rounded_binary64(value)
        error = abs(Fraction.from_float(result) - value)
        self.counts['fallback_' + operation] += 1
        self.labels[label] += 1
        self.counts['inexact'] += int(error != 0)
        self.counts['exact_nonzero_below_min_subnormal'] += int(0 < abs(value) < MIN_SUBNORMAL)
        self.counts['exact_nonzero_below_min_normal'] += int(0 < abs(value) < MIN_NORMAL)
        self.counts['rounded_zero'] += int(result == 0)
        self.counts['rounded_nonzero_subnormal'] += int(0 < abs(result) < float.fromhex('0x1p-1022'))
        if error > self.max_error:
            self.max_error = error
            self.max_error_case = {'label': label, 'operation': operation,
                                   'a_hex': a.hex(), 'b_hex': b.hex(),
                                   'result_hex': result.hex(),
                                   'exact_value': fraction_record(value)}
        if len(self.examples) < 12:
            self.examples.append({'label': label, 'operation': operation,
                                  'a_hex': a.hex(), 'b_hex': b.hex(),
                                  'result_hex': result.hex(),
                                  'rounding_error': fraction_record(error)})
        return result

    def summary(self):
        count = self.counts['fallback_multiply'] + self.counts['fallback_divide']
        return {'scope': 'Products/quotients inside normalized() only; no error suppression or floor',
                'rounding': 'Integer rational nearest binary64 with ties to even; bit construction',
                'counts': dict(self.counts), 'labels': dict(self.labels),
                'maximum_absolute_local_rounding_error_exact': fraction_record(self.max_error),
                'sum_absolute_local_rounding_errors_upper_bound': fraction_record(count * self.max_error),
                'maximum_error_case': self.max_error_case, 'first_fallback_examples': self.examples,
                'error_bound_scope': 'Local fallback roundings only, not a propagated output/solve error certificate',
                'contraction_scope': 'Tiny contractions use fixed ascending index multiply/add order; ordinary einsum/dot graph identity is not asserted for fallback contractions'}


class NormalizationArithmetic(TinyArithmetic):
    """NumPy adapter; NumPy is passed by an already admitted caller."""
    def __init__(self, numpy):
        super().__init__()
        self.np = numpy

    def binary(self, a, b, operation, label):
        np = self.np
        aa, bb = np.broadcast_arrays(np.asarray(a, dtype=np.float64),
                                     np.asarray(b, dtype=np.float64))
        mask = np.empty(aa.shape, dtype=bool)
        for i in np.ndindex(aa.shape):
            mask[i] = possible_tiny(aa[i], bb[i], operation)
        self.counts['candidate_entries_' + operation] += aa.size
        ufunc = np.multiply if operation == 'multiply' else np.divide
        if not bool(np.any(mask)):
            return ufunc(aa, bb)
        out = np.empty(aa.shape, dtype=np.float64)
        ordinary = ~mask
        # Tiny entries are excluded before the ufunc, not evaluated then hidden.
        out[ordinary] = ufunc(aa[ordinary], bb[ordinary])
        for i in np.ndindex(aa.shape):
            if mask[i]:
                out[i] = self.exact(aa[i], bb[i], operation, label)
        return out[()] if out.ndim == 0 else out

    def mul(self, a, b, label):
        return self.binary(a, b, 'multiply', label)

    def div(self, a, b, label):
        return self.binary(a, b, 'divide', label)

    def contract(self, a, b, label, vector=False):
        np = self.np
        aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if aa.ndim != 2 or bb.ndim != (1 if vector else 2) or aa.shape[1] != bb.shape[0]:
            raise ValueError('normalization contraction shape differs')
        cols = 1 if vector else bb.shape[1]
        tiny = False
        for i in range(aa.shape[0]):
            for j in range(cols):
                for k in range(aa.shape[1]):
                    right = bb[k] if vector else bb[k, j]
                    tiny = possible_tiny(aa[i, k], right, 'multiply') or tiny
        if not tiny:
            self.counts['ordinary_vector_contractions' if vector else 'ordinary_matrix_contractions'] += 1
            return aa.dot(bb) if vector else np.einsum('ik,kj->ij', aa, bb, optimize=False)
        self.counts['fallback_vector_contractions' if vector else 'fallback_matrix_contractions'] += 1
        result = np.empty((aa.shape[0], cols), dtype=np.float64)
        for i in range(aa.shape[0]):
            for j in range(cols):
                value = np.float64(0)
                for k in range(aa.shape[1]):
                    right = bb[k] if vector else bb[k, j]
                    product = self.mul(aa[i, k], right, label + ':product')
                    value = np.add(value, product)
                result[i, j] = value
        return result[:, 0] if vector else result

    def mm(self, a, b, label):
        return self.contract(a, b, label)

    def mv(self, a, b, label):
        return self.contract(a, b, label, vector=True)
