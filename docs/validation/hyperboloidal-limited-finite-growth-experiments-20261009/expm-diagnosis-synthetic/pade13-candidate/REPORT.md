The original limited J0/N8 growth attempt remains failed.  A fresh isolated
call reproduces its FloatingPointError in SciPy's pick_pade_structure,
_matfuncs_expm.pyx line 128.  The saved argument is finite.  Separate product
probes also raise divide-by-zero for synthetic 64x64 identity/identity and
zero/ones products, even though the returned out arrays are finite and bitwise
equal to literal einsum.  This supports a library/backend floating-status
problem as an explanation to investigate; it identifies no exact C-level
cause and supplies no propagation acceptance.  All exceptions and original
receipts remain preserved.  The optional-PyYAML UserWarning from np.show_config
is recorded separately from the numerical error.

The fresh helper is real binary64 fixed [13/13] Padé scaling and squaring.
For B=2^(-s)A, s is the smallest nonnegative integer satisfying
||B||_1 <= 5.371920351148152, with a check for rounded-log boundary effects.
It forms B2, B4, B6, then the published factored odd/even polynomials U and V,
solves (V-U)R=V+U, and squares R exactly s times.  All matrix products use
np.einsum('ik,kj->ij', optimize=False); only the rational linear system uses
numpy.linalg.solve.  There is no inverse, balancing, eigenvalue calculation,
adaptive threshold, fallback, warning suppression, or clipping.  Intermediate
underflow/overflow and nonfinite arithmetic are errors.  See PRIMARY-SOURCES.md
for the distinction from the later adaptive Algorithm 5.1 and its theta=4.25.

Before running the gates, PLAN.md fixed forward error 2e-11, rational solve
residual 2e-14, oracle agreement 1e-85, and bounded consistency error 5e-11.
Exact Fraction arithmetic independently recovers all integer coefficients
from the factorial formula and verifies 27 Taylor coefficients through degree
26.  Static AST checks exclude @, dot, matmul, and optimized einsum products.

The first synthetic gate passed 23 cases, including 18 independent mpmath
exponentials at 100 and 130 decimal digits.  Zero/empty, diagonal, rotation,
Jordan, dense, exact-rational nonnormal similarity, non-diagonal SPD energy
similarity, scaling-boundary and dimension-64 cases are included.  The largest
forward error was 7.274838030398736e-15, the largest normalized rational solve
residual 7.875572349800863e-17, and the largest bounded inverse consistency
error 9.360830777728722e-14.  Five invalid input families were rejected.

A separately declared four-case large-norm supplement passed the same forward
and solve thresholds and independent 100/130-digit oracles.  It reaches norm
131072 and 15 squarings; the largest forward error was
3.637978807091713e-12.  These are intentionally nonnormal examples; their
forward errors do not establish a bound for other matrices.  The separately
named same-failure-runtime check passed all 27 constructed cases, with outputs
bitwise equal across both Python environments.  NumPy all='raise' and warnings
as errors were enabled, stricter in underflow than the original analyzer.
All three synthetic command receipts have returncode zero and empty stderr.
No saved scientific matrix was loaded or evaluated by the helper gates.

The parent reviewed the helper formulas against the primary author paper with
no correction.  A separate independent static review of its new analyzer
copy confirms the complete numerical try/except body is AST-identical to the
failed original, as are all old non-analyze helpers.  Only the exponential
binding, mandatory helper/addendum admission pins, and diagnostic bookkeeping
change.  The wrapper's existing 2e-9 algebra threshold checks a rational solve
residual, not exponential forward error.  Original times, E transform,
spectrum, seeds, half-time consistency, short RK3 and guards are unchanged.

These gates admit consideration of a separately released finite-matrix replay
only.  They do not admit a continuum, native Cartesian, nonlinear, scri or
black-hole stability conclusion.  Full nongauge continuum comparison remains
unresolved.  No scientific replay has been performed by this review package.
