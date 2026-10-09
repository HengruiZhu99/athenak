The separately authorized actual J0/N8/rb=.98 accuracy check at t=6 passed.
The exact existing mm/energy_transform ASTs reconstruct binary64 A,L,J without
importing or executing analyze(); all three arrays are bitwise identical to
the earlier isolated-input snapshot.  The passing root receipt, payload,
matrix, analyzer, and unchanged Padé helper are hash-bound before preparation.
No generator eigensolve or PDE/API query is performed here.

The independent reference uses mpmath1.3.0's Taylor-series matrix exponential
at100 and130 decimal digits, treating the binary64 A entries as exact and
multiplying by the exact integer6.  The full propagator, energy seed states,
and modal seed states agree between precisions on normalized peak scales
2.7494e-102,3.8168e-102,1.2132e-101 respectively.  Both complete decimal
oracles are saved.  The modal conversion is an independently implemented
high-precision triangular back substitution; initial energy seed entries
are the exact binary64 values consumed in the parent algorithm.

The floating helper instead first rounds6*A in binary64, as the parent does.
The largest argument-rounding difference is9.094947017729282e-13.  The
comparison therefore includes that rounding as well as floating exponential
and seed-action errors; no entrywise rounding difference is hidden.

The full floating propagator differs from the130-digit reference by
5.199253160215285e-6 in Frobenius norm, scaled4.382086678594264e-12, and
1.400592736899853e-6 in peak absolute error, scaled4.743609029944668e-12.
All eight saved final modal seed columns combined differ by
1.596298453887515e-8 in Frobenius norm, scaled4.582069855383735e-12.
The largest separately scaled column discrepancy is1.585011413089253e-11
in energy coordinates and1.363305532452780e-11 in modal coordinates.  Every
check passes the predeclared2e-7 threshold; no threshold or source changed
after execution.  Scales use max(1,reference norm,computed norm), or
max(1,reference peak), as explicitly recorded in source.

The helper uses14 squarings, scaled one norm4.186049086560831, and rational
solve residual9.460267787534894e-17.  The command took19.22337725 seconds,
returned zero and emitted empty stderr, with warnings as errors and NumPy
all='raise'.  High-precision calls took8.983 and10.102 seconds.

This independently supports the numerical accuracy of the one finite-matrix
t6 result.  The parent's bitwise-zero half-time product discrepancy is only
an internal consistency check: its fixed scaling/squaring reuses nested
operations and is not independent accuracy evidence.  The strong finite
growth is not a continuum/native/nonlinear/scri/black-hole stability result.
Full nongauge continuum comparison remains unresolved.  Both failed ordinary
FD gates and the original SciPy attempt remain failed and unchanged.
