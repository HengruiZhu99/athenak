# Candidate 2 local qualification

Code under test: `9f64fcf6113ac9efd31b2957d1677d8203f5ebca`.

This candidate replaces the analytic production primary kernel with four
ordinary flat-kernel stages:

1. 32-Real active-cell physical geometry plus the gauge-driver RHS;
2. 64-Real packed inverse-Psi/q/Delta/gauge-source preparation;
3. ten component-parallel covariant scalar sources;
4. ten component-parallel Pi principal updates.

The transient allocation is exactly 106 Reals per active cell, split 32+64+10.
The accepted persistent analytic reference allocation remains 12+8+141 Reals
per ghosted cell, and analytic mode allocates zero generic-cache bytes.

The unchanged source-unit run passed the deterministic coefficient, expanded
radial, generated-geometry, mixed moving-gauge/dtTheta, compact-boundary, and
all-61 gates.  The 4320-sample compatible/standard all-61 error remained
`2.84217e-14` against the binary64-conditioned tolerance.

`evolved/` contains fresh one-cycle analytic and generic-cache histories for
both compatible and standard Phi ordering.  Both comparisons pass the
unchanged `5e-12` threshold with overall conditioned Linf
`2.9721190786258234e-13`; common-ADM histories agree to `6.54e-15` or better.
Restart files were deliberately excluded.

This is local correctness evidence only.  PVC full-output execution, compiler
pressure, and matched performance remain unqualified at this checkpoint.
