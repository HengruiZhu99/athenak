The candidate uses fixed real binary64 degree 13, norm-one scaling with the
published theta13=5.371920351148152, a direct rational linear solve, and literal
unoptimized einsum for every matrix product and every squaring.  It is a fresh
implementation of equations (3.5)--(3.6) and the scaling step of Algorithm 3.1
in Al-Mohy and Higham (2009), not the improved Algorithm 5.1.  There is no
adaptive parameter, tolerance-dependent algorithm, or warning suppression.

Before execution, the synthetic gate thresholds are fixed as follows:

- Exact Fraction arithmetic verifies every integer coefficient independently
  from the factorial formula and the degree-26 Taylor matching conditions.
- All numerical warnings become exceptions; NumPy all='raise' is retained.
- Synthetic finite outputs must match closed-form/high-precision outputs with
  max-entry error divided by max(1,max-absolute-reference) <= 2e-11.
- The normalized rational solve residual must be <= 2e-14.
- Independent mpmath expm oracles at 100 and 130 decimal digits must agree
  to relative/max-entry scale 1e-85 before rounding to float64.
- Synthetic half-time composition and inverse checks use a declared 5e-11
  threshold on examples with bounded exp(A) and exp(-A).  These are extra
  consistency checks, not independent exponential accuracy bounds.
- Cases cover empty/zero/identity, diagonal growth/decay, nilpotent Jordan,
  rotations, fixed dense real matrices, an exact-rational nonnormal similarity,
  an independently chosen non-diagonal SPD energy similarity, both sides of
  theta13 and power-of-two scaling boundaries, and dimension-64 examples.
- Invalid non-square/complex/nonfinite inputs must raise.  AST inspection must
  exclude @, matmul, dot, scipy, and optimized einsum in the helper.
- All constructed input/output arrays and high-precision oracle strings are
  saved.  No scientific operator is loaded or evaluated by these gates.

The initial SciPy growth failure and all diagnostic products remain unchanged.
Passing this gate only supports a further independent source review.  Actual
finite-matrix replay remains held until the parent explicitly releases it.
