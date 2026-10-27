# Initial-background derivative bases: controlled and unadopted

These new standalone probes use the frozen supported-v15 equations, the selected
M=1/3 isotropic wormhole initial data and the Minkowski reference. Physical
unknown values are unchanged. No physical RHS residual is subtracted, no shift
is pinned, and the puncture/core discretization is untouched. They are
transition-only derivative hypotheses, not coupled evolutions or a generic
wormhole-to-trumpet qualification.

The known analytic initial profile B is sampled on actual active donor points.
For chi, q=log(chi); for lapse, q=log(alpha/Aref). The profile is a basis function
only; its coefficient is free per field/stencil, never prescribed, stored or
evolved. No taper multiplies B: the local derivative probe is evaluated only at
transition queries. Therefore no taper derivatives have been omitted. A future
implementation that localizes the function itself must include all derivatives
of that localization. No singular-power assumption is imposed on the evolving
core.

## Fixed DB plus finite differences of the deviation fails constants

For a fixed analytic background,

    D_fixed q = D B + D_h(q-B).

This restores initial derivatives by construction, but

    D_fixed 1 = D B - D_h B

is generally nonzero. At point126, the chi constant-control gradient/Hessian
component defects are .0142134/.331937. The ordinary constant-control errors are
zero/1.04e-14. The fixed decomposition is therefore not admitted as a universal
derivative operator or silent Minkowski-preserving improvement.

Exact initial derivatives do not make the actual physical state stationary:
original full tensor/gauge rates at point126 are

| Physical rate | Value |
|---|---:|
| chi_t | -.0424355 |
| Khat_t | -.0386849 |
| alpha_t | -1.42411 |

H is approximately-8e-16 there. Every displayed rate is computed from the
original physical RHS; none is canceled. For a matched strong nonspherical
perturbation, fixed decomposition can also worsen lapse-source error. Initial
accuracy alone is insufficient.

## Free-amplitude P5 plus B

The alternative keeps all56 Cartesian monomials through degree5 and adds the
analytic initial B column. Its amplitude is fitted freely. Constants/P5 and B
can then be represented simultaneously without forcing future states to equal
B. The field value at the query is always the actual physical/stored value,
never the fitted value. Initial/scaled-B, constant/P2, small smooth, angular,
strong nonspherical and changed radial profiles are tested with seeded noise.

The first48-query,288-plan experiment keeps rank/condition/reproduction gates
and physical value/gradient/Hessian functional L1 caps16/32/64. It separately
predeclares a raw coefficient-L1 cap10000. Only12 plans pass all gates;276
reject explicitly. Point126 chi/256 reproduces constants/P2 and initial/scaled
B to approximately1e-13, and initial H error is6.86e-15. Nevertheless this is
incomplete geometry coverage, not an all-domain result or production candidate.

The original rejection categories are preserved, including overlaps:

| Disjoint original failure reason | Count |
|---|---:|
| P5/background rank failure |24|
| Early weight guard, components not recoverable after cleared rows |36|
| Coefficient cap alone |11|
| Physical and coefficient caps |11|
| Reproduction and coefficient caps |181|
| Reproduction, physical and coefficient caps |13|

Across overlapping categories,194 fail reproduction,24 physical amplification
and216 coefficient amplification. The early36 are not relabeled from missing
post-failure weight rows. A compile-only nested-main helper error precedes this
qualified run and remains separately recorded.

## Independent normalization check

Replacing B by10B leaves the function span and physical unknown unchanged. The
same coefficient cap changes status because the nuisance coefficient transforms
inversely. At point126 alpha/256, coefficient L1 changes18038.6→1803.86, making
that particular cap switch from reject to pass while physical derivative weights
stay unchanged to roundoff. The original288 results remain frozen; this new
experiment does not rewrite them as passing.

Across228 comparable built plans, maximum dimensionless physical-functional
change is1.835e-11, below the predeclared1e-9 check. Matched qualified full-source
errors change by at most3.85e-14. With the same raw threshold, ready counts are
12/288 for B and40/288 for10B. Raw coefficient amplification alone is therefore
not a normalization-invariant stability criterion. Reproduction diagnostics also
have basis/functional units; the old1e-10 gate is retained rather than silently
lowered.

Matched strong nonspherical source controls at point126 are:

| Control | Chi full-source error max | Lapse full-source error max |
|---|---:|---:|
| Original FD4 |.0130440|.00971175|
| Fixed DB+FD deviation |.00790919|.0164502|
| Qualified free basis (same physical span) |.000851195|.00175525|

For chi, H error decreases from-.0395032 to-.00156095. These finite errors are
retained; the test does not prove generic profile robustness or late collapse.

## P5-orthogonal background residual: same span, better numerical basis

A separate handful-query audit first QR-projects the known B column onto the
complete P5 space. It then replaces that column by

    B_res = (B - P5_fit(B)) / weighted_residual_norm.

Its query value/gradient/Hessian subtract the same polynomial and divide by the
same norm. All P5 modes and a free residual amplitude remain, so this is a basis
transformation, not a change of function span, physical field, PDE or source.
A rank-deficient P5 geometry is rejected before this operation; it is never
repaired by dropping columns.

Actual Apple ARM precision is recorded: `sizeof(double)=8`,
`numeric_limits<double>::digits=53`, and the same8bytes/53digits for long double.
No extended precision is assumed or used.

On four queries, two fields and three supports (24 plans), all unchanged gates
pass for16 preconditioned plans versus2 original plans. Four P5 rank failures
and four physical-amplification failures still reject. At point126/256:

| Estimate | Chi | Lapse |
|---|---:|---:|
| Original57-column QR condition |39201.6|8088.58|
| Preconditioned57-column QR condition |1131.46|1131.43|
| Preconditioned reproduction defect |1.07e-13|2.07e-13|

Original, P5-only and preconditioned estimates are retained separately. A smaller
preconditioned estimate is not claimed to change the conditioning of the original
physical problem. Maximum scaled physical-weight change is7.92e-12 and maximum
matched full-source change5.82e-13. Constants/P2, initial/scaled B, angular,
changed-radial and strong nonspherical controls remain; seeded scaled noise is
at most3.68e-12. Active values are never projected.

This removes a numerical basis-conditioning/reproduction artifact on the tested
geometries. It does not improve the exact-arithmetic stencil operator or erase
its physical approximation errors. Full transition coverage, background-oracle
uncertainty controls and coupled scientific gates remain required before any
adoption.

## Guard interpretation and next gates

Physical functional amplification and measured physical-source/noise sensitivity
are meaningful for the derivative operator. QR/reproduction diagnostics must be
stated with their dimensionless basis conventions. A raw nuisance coefficient
bound becomes meaningful only together with a corresponding, normalization-
invariant bound on uncertainty in its background function and query jets.
No existing physical/rank/condition/reproduction gate is relaxed here.

Required follow-ups include complete transition geometry coverage; independent
background value/jet uncertainty; matched strong nonspherical/profile tests;
and all-cell coupled residual, boundary and radial-comparison gates. Neither a
successful initial solve nor these small probes establish generic regularity,
core motion, a stationary wormhole or a completed black-hole benchmark.

## Frozen evidence

- `transition-background-basis-20261027`: compile-only helper failure.
- `transition-background-basis-r2-20261027`: original288 plans/cap10000,
  constant defect and nonzero original physical rates.
- `transition-background-normalization-20261027`: independent B/10B experiment;
  original failures untouched.
- `transition-background-analysis-20261027`: exact rejection accounting,
  matched source errors and inspected plot.
- `transition-background-orthogonal-20261027`: four-query same-span audit,
  precision receipt, original/preconditioned estimates and controls.
- `transition-background-orthogonal-analysis-20261027`: guard/covariance analysis
  and inspected plot.

All use frozen source/initial/oracle hashes, at most180seconds per newly owned
CPU process, and no GPU/MPI/remote allocation or original-job modifications.
Source capsules are archived and raw evidence remains intact. Fresh directories:

```
python3 tst/elliptic_shift_3d/run_transition_background.py --output NEW_BG
python3 tst/elliptic_shift_3d/run_background_normalization.py --output NEW_NORM
python3 tst/elliptic_shift_3d/analyze_transition_background.py --original NEW_BG --normalization NEW_NORM --output NEW_BG_ANALYSIS
python3 tst/elliptic_shift_3d/run_background_orthogonal.py --output NEW_ORTH
python3 tst/elliptic_shift_3d/analyze_background_orthogonal.py --probe NEW_ORTH --output NEW_ORTH_ANALYSIS
```

The runners intentionally bind the earlier frozen v15 source/oracle capsule;
relocation must preserve or explicitly update that recorded path. No production
engine/checkpoint/AF-sampler/core-map change is part of this work.
