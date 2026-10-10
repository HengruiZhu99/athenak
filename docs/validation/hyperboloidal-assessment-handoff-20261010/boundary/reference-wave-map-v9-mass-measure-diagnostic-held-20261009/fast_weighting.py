"""SOURCE-ONLY whole-array normal-product proof; NumPy supplied by admitted caller."""
from collections import Counter
import math
from tiny_normalization import NormalizationArithmetic


class FastWeightingArithmetic:
    def __init__(self, numpy):
        self.np = numpy
        self.fallback = NormalizationArithmetic(numpy)
        self.counts = Counter()
        self.labels = Counter()
        self.minimum_proven_exponent_sum = None
        self.first_fallback_bounds = []

    def mul(self, a, b, label):
        np = self.np
        aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        # Check the original operands, including a nonfinite scalar broadcast
        # against an empty array: no invalid operand is hidden by broadcasting.
        if not (bool(np.isfinite(aa).all()) and bool(np.isfinite(bb).all())):
            raise ValueError('finite binary64 weighting operands required')
        aa, bb = np.broadcast_arrays(aa, bb)
        self.counts['calls'] += 1
        self.counts['broadcast_entries'] += aa.size
        self.labels[label] += 1
        if aa.size == 0:
            self.counts['proved_empty_calls'] += 1
            return np.multiply(aa, bb)
        az, bz = np.abs(aa), np.abs(bb)
        an, bn = az[az != 0], bz[bz != 0]
        if an.size == 0 or bn.size == 0:
            self.counts['proved_all_zero_products_calls'] += 1
            self.counts['proved_zero_entries'] += aa.size
            return np.multiply(aa, bb)
        ea = math.frexp(float(np.min(an)))[1]
        eb = math.frexp(float(np.min(bn)))[1]
        total = ea + eb
        if total >= -1020:
            self.counts['proved_normal_or_zero_calls'] += 1
            self.counts['proved_normal_or_zero_entries'] += aa.size
            if self.minimum_proven_exponent_sum is None or total < self.minimum_proven_exponent_sum:
                self.minimum_proven_exponent_sum = total
            # Exactly the original ufunc: overflow/invalid still raise normally.
            return np.multiply(aa, bb)
        self.counts['delegated_fallback_calls'] += 1
        self.counts['delegated_broadcast_entries'] += aa.size
        if len(self.first_fallback_bounds) < 12:
            self.first_fallback_bounds.append({'label':label,
                'minimum_nonzero_frexp_exponents':[ea,eb],
                'sum':total,'normal_proof_requires_sum_at_least':-1020})
        # Same unchanged per-pair predicate and exact helper as v3.
        return self.fallback.mul(aa, bb, label)

    def summary(self):
        result = self.fallback.summary()
        result['fast_weighting_proof'] = {
            'counts':dict(self.counts),'labels':dict(self.labels),
            'minimum_proven_exponent_sum':self.minimum_proven_exponent_sum,
            'first_delegated_bounds':self.first_fallback_bounds,
            'proof':'Every nonzero |a|>=2^(ea_min-1), |b|>=2^(eb_min-1). Sum>=-1020 implies every nonzero product>=2^-1022; possible_tiny is false for every pair.',
            'scope':'Sufficient whole-array proof only; inconclusive arrays delegate unchanged, including disjoint tiny nonzero entries whose actual products are all zero.',
            'overflow':'Normal lower bound does not bound maximum products; original strict NumPy overflow behavior retained.'}
        return result
