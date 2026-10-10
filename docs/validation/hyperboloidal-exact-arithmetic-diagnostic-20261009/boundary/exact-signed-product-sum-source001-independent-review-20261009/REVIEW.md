The complete arithmetic header, probe, Fraction oracle, runner, registry and
plan have no mathematical blocker found by this source review. This is a
source argument, not an executed unit result or an exhaustive proof of compiler
behavior. No candidate import, Fraction target evaluation or compiler ran.

Every finite atom is exactly m*2^e with m<2^53, e in[-1074,971]. Four factors
fit212 significand bits. Each word-by-significand multiply plus carry stays
below118 bits, within unsigned128; the four-word product fits256 bits. The
lowest included exponent is-4297, above accumulator base-4352. At most128
monomials per signed accumulator have magnitude<2^4103, below its upper
capacity2^4352. AddWord's carry propagation and split AddProduct add the exact
shifted product; compare/subtract uses one unsigned borrow chain after complete
positive/negative accumulation. Validation visits every consumed atom before
zero terms are omitted. Unused atom slots are explicitly outside the domain.

Round's exact maxfinite comparison enforces the stronger declared domain,
including maxfinite-plus-tiny rejection. Highest/trim retain53 normal bits or
the fixed subnormal grid. Guard plus sticky and retained parity implement
nearest-even; carry handles a binade transition or minimum-normal correctly.
The domain precheck prevents infinity. Exact cancellation is+0; nonzero
negative underflow may be-0. The integer half-grid exponent bounds only local
rounding. Rejected bits are sentinel0, never an accepted numeric result.

EvaluateDual validates value and tangent independently, then replaces each
factor once without division, deduplication or seed recognition. All arities
are<=4 and at most32 inputs, so the complete128 product-rule terms fit. Zero
primal/nonzero tangent survives. Primal and derivative may overflow separately.

The Fraction oracle is algorithmically independent: exact formal polynomial
recurrence and CPython rational conversion rather than fixed limbs/guard bits.
Its expanded product list only checks counters; the separate recurrence checks
the derivative target. Literal hand bits cover signed zeros, normal/subnormal
ties, binade carry and cap edges. The70 registry cases exactly match cases.txt;
44 scalar/26 dual counts and the32x4=128 case are source metadata, not targets
evaluated in this review. Invalid scalar status is checked, while sentinel-bit
coverage for invalid input is primarily inspected in header source; overflow
sentinels are explicitly oracle-gated. This is not a new universal error bound.

Admission has one blocker: recipe selects the literal clang invocation, while
the C++ probe uses iostream/string/vector and no explicit C++ driver mode or
stdlib link flag. The resolved binary may be shared with clang++; the invoked
driver name still matters. Root confirmed this finding and requested a fresh
source002. The upstream LLVM driver source shows clang++ selecting g++ mode
and C++ standard-library linking conditioned on CCCIsCXX:
[LLVM ToolChain.cpp](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChain.cpp).
This source citation supports the driver distinction; no local compiler probe
was executed. Source001 remains held and is not labeled an actual compile FAIL.

The runner otherwise requires a passed review bound to its exact source index,
fixed isolated unoptimized Python/env, pre/post source/runtime/header hashes,
unique attempt, MD dependency subset, Release+ASan/UBSan outputs and exact
oracle/byte equality. These are future gates. This primitive cannot recover
already rounded inverse/determinant/auxiliary atoms and does not repair any of
the original63 far-dual failures.
