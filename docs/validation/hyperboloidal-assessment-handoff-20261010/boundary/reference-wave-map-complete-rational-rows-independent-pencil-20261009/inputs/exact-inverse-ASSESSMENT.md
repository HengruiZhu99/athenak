# General exact3x3 first-dual inverse: mathematical proposal only

No inverse is implemented here. No numerical arithmetic, CAS, compilation,
query, target generation or matrix evaluation is released. The standalone
four-factor signed-product primitive and its held unit suite remain a
separate proposal. The original63 far-dual failures remain FAILED.

## Meaning of the proposed output

Let A and E be arbitrary real3x3 matrices whose18 submitted entries are finite
binary64 numbers, interpreted as exact dyadic reals. E is the supplied first
variation of A. Do not assume symmetry, positive definiteness, determinant1,
trace-free E, a known seed, or a diagonal/reference metric. If det(A)=0 exactly,
the inverse and its derivative are undefined and must be rejected explicitly.

For det(A)≠0 the targets are

    G = A^−1,
    G_dot = −G E G.

Each of these18 real entries is rounded ONCE to binary64 at final output.
The dual target is the analytic derivative of the real inverse at the exact
submitted A, not the derivative of the discontinuous binary64 rounding map.
This matches the intended mathematical first-dual operation. It is not the
derivative of an inverse that has already been rounded and then treated as
exact. Zero primal cofactors or entries never remove a potentially nonzero
variation. A full dual validation checks every consumed E entry even when
the corresponding primal input is zero.

A proposed contract consistent with the short-sum primitive rejects any
individual exact output magnitude above maxfinite before rounding; entries
that round to a subnormal or signed zero remain permitted. This stronger
exact-domain policy must be explicit if selected. Exact singularity is a
separate status, with no determinant epsilon, positive floor or fallback
identity. Downstream users must inspect all statuses. Existing physical SPD
and geometry checks remain separate; algebraic invertibility is not an SPD
proof or a physical admissibility proof.

## Exact common-scale integer construction

Every finite binary64 value can be written as an integer times2^−1074. Write

    A = 2^−1074 B,      E = 2^−1074 F,

where B and F are signed integer matrices, each magnitude strictly below
2^2098. This representation is exact even for subnormals, zeros and extreme
normal inputs. It is deliberately simple; later implementation may reduce
common powers of two, but no such optimization is needed for the argument.

Let D=det(B), and C be the cofactor matrix of B, so adj(B)=C^T. Compute all
two-product cofactors and the six-product determinant with signed integer
arithmetic. Their magnitudes obey

    |C_ij| < 2^4197,       |D| < 2^6297.

These are conservative bounds, not floating intermediates. Exact cancellation
may leave a very small nonzero D; it must not be classified by a rounded
determinant. Determinant/cofactor overflow or underflow in ordinary binary64
is irrelevant to this exact integer stage.

For D≠0,

    G_ij = 2^1074 C_ji / D,
    (G_dot)_ij = −2^1074 (C^T F C^T)_ij / D^2.

The latter numerator is a nine-term sum of cofactor*variation*cofactor. Each
cofactor has two degree2 terms, so expanding it gives at most36 degree5
monomials per output. This is a complete polynomial identity, with no seed
or determinant-derivative assumption. Equivalently,

    G_dot = 2^1074 (C_dot^T D − C^T D_dot) / D^2,
    D_dot = sum_ij C_ij F_ij.

Here C_dot differentiates the cofactors of B in direction F; generally
D_dot is NONZERO. The two forms provide an independent algebraic cross-check.
The first avoids forming separately rounded C_dot,D_dot or ratios.

The entry H=(C^T F C^T)_ij has |H|<2^10496. Thus one unreduced rational
representation has primal numerator below2^5271 and denominator below2^6297,
and tangent numerator below2^11570 and positive denominator below2^12594.
Signs are kept separately and the primal denominator is normalized positive.
These prove finiteness of the required integer workspace. They do not fit the
existing four-factor/32-monomial public primitive: degree5 and36 terms exceed
its declared bounds. Do not silently enlarge or reuse its rounded output.
A later inverse implementation requires its own capacity/long-division proof,
source and unit review. No fixed inverse limb allocation is selected here.

## Correct rounding of the exact rational entries

For a positive denominator and signed numerator N, handle N=0 as canonical+0.
Before rounding, compare |N| with maxfinite*denominator exactly if using the
stronger exact-output domain. For nonzero admitted magnitude x, determine
floor(log2(x)) by integer bit lengths and exact cross-multiplication; no
floating logarithm or approximate reciprocal is required.

Choose grid exponent e=floor(log2(x))−52 when x≥2^−1022, or e=−1074 below
minimum normal. Form x/2^e as a ratio of integers by shifting the numerator
or denominator. Exact integer quotient/remainder division yields Q,R with
0≤R<denominator_effective. Round upward precisely when

    2R > denominator_effective, or
    2R == denominator_effective and Q is odd.

Normalize a normal53-bit carry, and allow a subnormal carry into minimum
normal. A nonzero negative magnitude rounding to zero produces−0. The
per-entry absolute rounding error is at most half this grid spacing; report
the spacing with integer exponent, not an underflowing floating estimate.
This is an integer rational rounder, not an FMA expansion or a binary64
division followed by an error correction. For the unreduced tangent bounds,
shifts at the subnormal grid require numerator width below12644bits; a normal
negative shift requires denominator width below13565bits. The exact maxfinite
comparison can require denominator*maxfinite width below13618bits. These
illustrate the additional workspace obligations without choosing a source
allocation or executing the algorithm.

## Independent future checks, if implementation is authorized

Use an exact Fraction Gauss--Jordan inverse with exact nonzero pivot selection,
structurally separate from a cofactor candidate. A formal first-dual rational
Gauss--Jordan calculation can independently check E variation; inverse of a
nonzero pivot pair(p,d) is(1/p,−d/p^2). Pivot selection may inspect exact primal
nonzero values but must not inspect a seed name or delete a zero-primal dual.
Target output bits may use CPython Fraction-to-float plus explicit hand ties,
as in the separately held short-sum suite. No targets are computed in this note.

The fixed registry should include arbitrary nonsymmetric and indefinite
invertible matrices, symmetric positive definite matrices with unrestricted
variations, sign/permutation/transpose identities, zero primal cofactors with
nonzero derivatives, repeated inputs and exact singular controls. Include
cases where ordinary determinants overflow or underflow while all inverse
entries remain representable, exact near-singularity, genuine output/tangent
overflow, and normal/subnormal/zero rational ties. For example a triangular
matrix with diagonal2^1023,2^52,1 and offdiagonal1 has an inverse offdiagonal
−2^−1075, requiring correctly rounded−0 despite an overflowing ordinary
determinant. Changing that offdiagonal to3 supplies the next subnormal tie.
The identity matrix with a nonzero offdiagonal E has a zero primal inverse
entry and nonzero derivative; a traceful diagonal E forbids assuming D_dot=0.

Check the18 output bits and statuses against exact targets, rather than
accepting only A*G≈I or a backward residual. Such residuals are useful separate
diagnostics but cannot certify every entry's rounding or every dual. Track
exact-domain overflow, singularity, signed zero and integer capacity failures
explicitly, without changing tolerances after evaluation.

## Limits for the actual source problem

This would exactly invert the matrix that is submitted, not recover a physical
metric before its construction/normalization rounded it. Ill-conditioning
still amplifies input perturbations and may produce legitimately huge G_dot.
An entrywise correctly rounded inverse does not make subsequent contractions,
connection differences, lapse/chi differences, poles or source assembly exact.
The26-context saved diagnostic independently identifies inverse contamination
and downstream arithmetic contributions. Both require complete end-to-end
gates; the short-sum primitive alone is insufficient. No promise is made that
this inverse proposal plus that primitive repairs all63 failures.

Any later integration must preserve the actual off-constraint physical-P/
Theta conventions, full nonflat reference connection, all fields and arbitrary
first-dual directions, with no STF-specific determinant shortcut. It requires
fresh source admission and the unchanged complete far-dual/finite-difference/
reference gates. No gauge adoption, continuum/native stability or black-hole
evolution claim follows from this mathematical proposal. The wormhole-to-
trumpet requirement with the Minkowski hyperboloidal reference remains intact.
