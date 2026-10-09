The parent separately authorizes one actual finite-matrix accuracy check:
J0/N8/rb=.98 at t=6, using the unchanged binary64 energy-coordinate generator
A=L^T J L^(-T), the actual physical seed columns, and the saved passing Padé
payload.  No PDE/API query, new generator spectrum, or other degree/radius is
admitted.  The earlier source/synthetic capsule remains immutable.

Prepare input using only the exact original mm/energy_transform function ASTs
with their original NumPy/SciPy dependencies; do not import/execute analyze().
Assert A,L,J are bitwise equal to the prior isolated-input snapshot and bind
the current operator, analyzer, successful receipt/payload, and helper hashes.

The primary oracle is mpmath.expm(6*A) at 100 and130 decimal digits, treating
the saved binary64 A entries as exact real numbers.  This differs explicitly
from first rounding6*A to binary64, as the floating helper does.  Save both
arguments and their difference; do not hide this distinction.  Two-precision
agreement must be <=1e-80 on max-entry relative scale before comparing to the
floating result.  The predeclared float comparison threshold is2e-7, matching
the existing finite-exponential consistency threshold, but now using an
independent high-precision oracle.  This checks accuracy for this one matrix;
it is not a general arbitrary-nonnormal error bound.

Save both full high-precision propagators and high-precision energy/modal seed
states as decimal strings.  Initial energy seed entries are the exact
binary64 values consumed by the parent algorithm.  Convert energy states to
modal states with an independently implemented high-precision triangular
back substitution using exact binary64 L entries.  Compare all8 saved final
seed states, each column separately, as well as the full floating propagator.
Use literal einsum for binary64 matrix products, warnings as errors, and
NumPy all='raise'.  No SciPy expm, generator eigensolve, or warning suppression.

This is only a limited finite-matrix diagnostic.  Both failed ordinary-FD
gates and the original SciPy attempt stay failed.  Full nongauge continuum
comparison remains unresolved; no PDE/native/nonlinear/scri/BH conclusion.
