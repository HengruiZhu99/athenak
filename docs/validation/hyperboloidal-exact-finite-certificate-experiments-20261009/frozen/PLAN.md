Exact finite-matrix eigenvalue certificate: source-only candidate
================================================================

Status: HELD. The first stage is source/proof review and synthetic tests only.
No actual N8/N12/N16 matrix certificate has been executed. The original SciPy
expm failure, ordinary-FD stops, accepted archives and production sources remain
unchanged. No source/gauge/boundary adoption is proposed.

Mathematical certificate
------------------------

J is a square complex matrix whose binary64 entries are treated as exact dyadic
numbers. V and W are arbitrary approximate square change-of-basis matrices,
also exact dyadic inputs. D=diag(saved lambda), N=WV and R=WJV-ND are evaluated
exactly, using integer arithmetic with one shared power-of-two denominator per
matrix. Define exact rational upper bounds

 nu >= ||I-N||_infinity, rho >= ||R||_infinity,
 epsilon = rho/(1-nu), provided nu<1.

The complex row bounds are max_i sum_j(|Re a_ij|+|Im a_ij|), which rigorously
bound the usual complex induced infinity norm. This deliberate conservatism
may make a valid enclosure too wide; it can never underestimate that norm.
No sqrt approximation or floating residual is used in these bounds.

If nu<1, the Neumann series makes N nonsingular with ||N^-1||<=1/(1-nu).
Since N=WV and the matrices are square, V,W are nonsingular; V^-1=N^-1 W.
Consequently the EXACT identity is

 V^-1 J V = D + N^-1 R,
 ||N^-1 R||_infinity <= epsilon.

Every eigenvalue lies in the union of closed discs B(lambda_i,epsilon).
For exact disc counting, connect two centers when their exact squared distance
is <=(2epsilon)^2. Touching discs must merge. Each component has strict positive
separation from every outside disc. To justify counts for the enlarged discs
centered at D rather than at the unknown perturbed diagonal, use the homotopy
D+s N^-1 R,0<=s<=1. Its ordinary Gershgorin discs lie inside this fixed union
for every s. Isolated components cannot exchange eigenvalues, and at s=0 a
component contains exactly its number of centers, with algebraic multiplicity.

A component entirely in Re z>0 (min Re lambda_i-epsilon>0, tested exactly)
therefore certifies that number of positive-real-part eigenvalues of J.
Likewise negative components can be counted. Overlapping components crossing
the imaginary axis remain unresolved. nu>=1, a wide enclosure or no positive
component is INCONCLUSIVE, not evidence of stability. No diagonalizability of
J, separation of repeated centers or forward-error theorem is presumed.

Exact outward arithmetic
-------------------------

All products, subtractions, norm bounds, divisions for epsilon, squared disc
distances and strict half-plane tests use arbitrary-precision Python integers
or fractions. Binary64 conversion uses exact as_integer_ratio. Fractions are
recorded as full integer numerator/denominator strings. For readable enclosing
binary64 endpoints, outward_binary64 finds the exact exponent and performs
integer divmod on the normal/subnormal grid. It returns directed endpoints,
including infinity if necessary; endpoint hex strings are supplemental only.
nearest_binary64 selects the nearest exact endpoint with a bit-parity ties-even
test. It rejects nonfinite-range endpoints in the target assembly. The proof
does not depend on displaying a rounded decimal or on the platform's current
floating-point rounding mode.

Synthetic gate (no actual matrices)
-----------------------------------

test_synthetic.py defines positive/negative diagonal, repeated positive-center,
complex-conjugate, overlapping, just-touching, strictly-separated, nearly
defective Jordan, bad-center, nu=1, singular-proposal, invalid input and extreme
dyadic-scale cases. Integer matrix products are checked against independent
Fraction loops. Directed rounding is checked at zero, normal ties-even,
subnormal half-ties, underflow, overflow and256 fixed-seed general rational
values. Each endpoint is compared to the exact rational; adjacent finite
endpoints are checked by their binary64 bit patterns. No actual file is read.
Tests must pass and be reviewed before any actual-matrix release. A failure is
retained in a fresh receipt/log, without changing tolerances or prior evidence.

Held actual input preparation
-----------------------------

run_saved_certificate.py is separately held by an exact per-N authorization.
It uses NumPy only to decode pinned NPZ files (allow_pickle=False), checking
finite shapes and lower Cholesky structure. No eigensolver or exponential is
called. Its target J is elementwise exact RN-even binary64(Jbulk+Jsat), matching
the previous analyzer's rounded binary64 target, not the unrounded dyadic sum
of the two stored matrices. The target's exact hex entries are preserved.
The preparer additionally requires equality with the current platform's direct
binary64 addition of those operands. This is a source/encoding binding check;
the mathematical proof uses the exact recorded target entries, not a claimed
interval for a different unrounded operator.

All full saved eigenvectors Q are in energy coordinates. The preparer computes
an ordinary binary64 PROPOSAL V=L^-T Q by triangular back substitution, and an
ordinary binary64 PROPOSAL W by complex Gauss-Jordan elimination. Neither
calculation is trusted for rigor: their resulting binary64 entries become exact
inputs, and the exact nu test alone proves invertibility and bounds the inverse.
Bad floating proposals may make the certificate inconclusive or fail preparation.
Saved eigenvalues are centers only; they are not assumed to be exact roots.

The final exact-binary64-input.json contains J,V,W,lambda as hexadecimal pairs;
certificate.json retains exact rational bounds, every component/count and strict
sign result. The receipt distinguishes completed computation from a successful
positive-root certificate. All output paths are fresh; input/source hashes are
recorded before and after. The input JSON is large_payload even if small.

Authorization keys: finite_matrix_exact_certificate_admitted=true,N,
driver_sha256,core_sha256,plan_sha256,operator_path/operator_sha256,
payload_path/payload_sha256,growth_receipt_path/growth_receipt_sha256,
synthetic_receipt_path/synthetic_receipt_sha256. The successful Pade receipt must
bind the same exact payload and operator. Synthetic receipt must bind the same
certificate core. First release will be only the synthetic test command; actual
matrix execution requires a later independent explicit release.

Scope and limits
-----------------

This can prove some positive eigenvalues of a declared rounded finite Galerkin
matrix. It does not prove positive eigenvalues of its unrounded quadrature
operator or the continuum PDE, classify a physical subsidiary mode, establish
constraint closure or identify a boundary cause. It is neither a native
evolution nor a nonlinear, exact-scri, CPBC or BH result. A uniform error radius
can be conservative for nonnormal bases, and preparing approximate inverses can
fail. Exact integer arithmetic can be slower than floating arithmetic; the
shared dyadic scale avoids Fraction arithmetic inside cubic matrix products.
N64/96/128 matrix dimensions are plausible but actual runtime is unmeasured.
Finite-pulse stability and later single-BH wormhole-to-trumpet formation with
the Minkowski hyperboloidal reference remain unresolved.
