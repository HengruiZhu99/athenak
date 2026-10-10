# Exact short signed-product sums: held standalone CPU unit proposal

This is a new arithmetic primitive, separate from the reference-wave-map gauge.
No gauge includes it. The saved far-dual gate remains FAILED, including all63
tangent failures. Exact sums of submitted binary64 atoms cannot undo errors
already present in a submitted inverse, determinant, auxiliary, or source term.
There is no kernel query, geometry replacement, seed-specific shortcut, FMA
optimization, operator, propagation, or production adoption in this proposal.

## Fixed domain and result contract

`signed_products::Evaluate(Term*,count)` accepts0..32 monomials. Each has
coefficient sign exactly+1 or−1, exponent shift exactly0 or−1, and0..4 finite
binary64 factors. A zero-factor product is the constant1. Unused atom slots do
not belong to the polynomial and are not inspected. All consumed atoms are
validated before any zero-product shortcut. A nonzero count requires a nonnull
pointer; zero count permits null. Invalid input returns a named status.

`EvaluateDual(DualTerm*,count)` has the same primal domain and validates both
value and tangent of every consumed atom. It constructs the complete ordinary
first product rule, with sum(arity)≤128 tangent monomials. It does not divide by
a primal factor, recognize a seed, discard a term because its primal is zero,
or deduplicate repeated factors. Even zero derivative monomials are counted.
Primal and tangent are summed and rounded independently. An invalid input has
one validation status; valid input may independently overflow either result.

For each exact real sum S, the admitted output domain is |S|≤maxfinite. An
exact larger magnitude returns `exact_overflow`, including maxfinite plus
half a minimum subnormal. This is intentionally stronger than merely rejecting
an infinite rounded result. Within the domain, output is correctly rounded
binary64 round-to-nearest/ties-to-even. Exact zero is+0. A nonzero negative
number that rounds to zero is−0. No positive floor, saturation, or silent
overflow substitute is present. Rejected result bits are a zero sentinel and
must never be consumed as an accepted numeric value.

## Integer inclusion and capacity proof

For a finite binary64 atom, decode sign, integer significand m and exponent e:
subnormal/zero m=frac52,e=−1074; normal m=2^52+frac52,
e=expfield−1075∈[−1074,971]. Thus the atom equals sign*m*2^e exactly,
with0≤m<2^53. Multiplying at most4 significands uses at most212 bits. The
four-word multiplication uses unsigned128 intermediates: a word*m plus its
carry fits in117 bits, well below128. All arithmetic before final encoding is
integer, including signs, exponent shifts and carry handling.

The minimum nonzero4-factor exponent including shift−1 is−4297. The two
136-word unsigned accumulators use BASE_E=−4352; their units therefore include
every admitted product, with55 low bits of slack at the minimum exponent.
Every atom magnitude is<2^1024, so every at-most4-factor monomial is<2^4096
(zeroarity constants also satisfy this). Each separately signed accumulator
has at most128 monomials, hence magnitude<2^4103. The array covers exponents
−4352..4351, so even this worst bound has ample high-bit slack. Positive and
negative monomials are accumulated separately; compare once and subtract the
smaller once. No rounded cancellation, intermediate floating overflow, or
floating underflow occurs. Unexpected limb overflow has an explicit failure
status despite being excluded by these domain bounds.

## Final rounding proof

Write the exact nonzero magnitude as integer N times2^BASE_E and find its
highest set bit b. Before rounding, compare N exactly with
(2^53−1)*2^(971−BASE_E), the maximum finite magnitude.

If E=BASE_E+b≥−1022, retain53 leading bits by trimming b−52 low bits. If
E<−1022, trim to the fixed2^−1074 subnormal grid. In either branch, the guard
bit is the highest omitted bit, and sticky is the OR of all lower omitted
bits. Increment exactly when guard&&(sticky||odd(retained)). A53-bit carry
renormalizes a normal result; a subnormal carry to2^52 encodes minimum normal.
Exact-domain comparison prevents a normal carry to infinity. A retained zero
keeps the sign of nonzero S; exact cancellation was handled earlier as+0.

The audit marks exact zero, inexact rounding, subnormal output and rounded
signed zero. `half_grid_exponent` is an INTEGER exponent describing an absolute
rounding bound2^(E−53) in the normal branch or2^−1075 in the subnormal branch;
it is not converted to an underflowing binary64 number. The audit is local to
this exact polynomial sum. It gives no bound for surrounding approximate
atoms or a larger gauge calculation.

## Fixed independent controls and acceptance

The immutable registry has70 cases:44 scalar and26 dual. `cases.txt` supplies
literal raw64-bit atom words to the C++ probe; each output echoes every input
word, including unused slots. The independent Python oracle reads the
separately pinned registry and requires exact echo/count/order agreement.

Targets use exact `Fraction.from_float` for finite submitted words. A formal
polynomial coefficient recurrence `(p,d)→(p*v,d*v+p*v_dot)` constructs the
first derivative independently from the C++ factor-replacement expansion.
The expanded Fraction list is used separately for zero/nonzero monomial
counters. CPython's exact-rational-to-float conversion supplies target bits,
which uses a different algorithm from the fixed-limb guard/sticky candidate.
Literal hand expectations additionally cover normal and subnormal ties,
binade carry, signed underflow-zero, exact cancellation, and degree/cap edges.
No tolerance or mpmath acceptance is used: result/status/bit/audit checks are
exact; the rounding bound is checked with exact Fraction arithmetic.

Coverage includes finite output after otherwise overflowing products;
subnormal output after otherwise underflowing products; a large exponent-gap
residue; four maximum and minimum atoms; separate positive/negative exact
overflow; stronger maxfinite-plus-tiny rejection; invalid atom/sign/shift/
arity/count/null inputs; zero primal with nonzero tangent; repeated and mixed
factors; independent primal/tangent overflow; and all128 generated monomials.
All finite-input ordinary paths use the SAME integer algorithm. There is no
special test-pattern branch. This finite unit suite is evidence for a reviewed
algorithm, not an exhaustive enumeration of all admitted bit patterns.

One fresh authorized attempt will compile standalone Release and ASan/UBSan
Debug C++17 CPU probes, run all70 controls in both modes, require zero stderr
and identical probe stdout bytes, and run the independent Fraction oracle.
Every command/stream/dependency/executable hash is retained in the attempt.
All sources, context, compiler/header/runtime and review pins are verified
before compilation and again after the attempt. Unknown compile dependencies
fail admission. No execution is released by this document or its readiness
metadata. The root must supply an exact source/recipe authorization and a
pinned passed source review. Existing attempt destinations are never reused.

## Scope and provenance

The integer argument is derived here, not borrowed as a floating-error
theorem. The prior combined-gradient pencil is context only. Its discussion
of Ogita/Rump/Oishi2005 and floating expansions does not certify this new
integer implementation. The fixed136-word CPU prototype uses unsigned
`__int128`; it is not a device/Kokkos routine or portable arbitrary compiler
claim. The preregistered header universe is a protected superset inherited
from the earlier standalone-capable CLT build closure, not evidence that this
new source has compiled. Actual `-MD` compiler dependencies must be a subset
or match a byte-exact copied local source. No compiler or numerical arithmetic
has run during preparation.
