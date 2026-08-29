# Ref-GH ordering, gamma2, and hyperbolic-gauge discriminator

Date: 2026-08-29 (America/New_York)

## Claim status

`CAUSE NOT YET ISOLATED`

The discriminator matrix has not yet run.  No standard-ordering puncture
repair, gauge-driver defect, gamma2/compatible-ordering defect, or common
Einstein/reference regression is claimed at this checkpoint.

## Exact checkpoint and historical control

- Parent branch: `codex/ref-gh-single-puncture-robustness-20260829`
- Parent commit: `39069a16d2d36e1bf5d124f7f274382eca4cd441`
- Discriminator branch: `codex/ref-gh-ordering-gauge-discriminator-20260829`
- Frozen production source: `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`
- Starting `git diff --exit-code a09caf70 -- src`: pass

The historical t=20M stationary campaign evolved 50 Einstein fields with
compatible Phi ordering, gamma0=1, gamma2 absent/effectively zero, no evolved
Hhat/theta/Upsilon driver, and KO=0.02.  Its exact stationary initial RHS was
of order 1e-16.

The failed robustness case evolved 61 fields with compatible Phi ordering,
gamma0=1, gamma2=1, the improved Hhat/theta/Upsilon hyperbolic gauge driver,
gauge-reference subtraction, and KO=0.02.  Its initial RHS Linf was
5.84253e-11, and it failed at cycle 330, t=1.123484M.  These are not identical
systems.

The committed run logs report the same initial maximum value and component
but two different radii: 2.49052M for the 216-rank run and 1.42324M for the
12-rank run.  The current initialization code globally reduces only the value,
not the associated location.  The location is therefore not yet globally
qualified; the Phase 2 diagnostic must repair this reporting defect without
changing evolution arithmetic.

## Phase 0 frozen regression gates

The fresh local gates passed without tolerance changes:

| Gate | Fresh result |
|---|---:|
| deterministic SymPy regeneration | two runs byte-identical to each other and committed headers |
| analytic coefficient oracle | 216 samples, max error 8.88178e-15 |
| expanded radial oracle | 2160 samples, conditioned error 1.48837e-13 |
| generated geometry oracle | 2376 samples, conditioned error 2.33147e-15 |
| moving gauge/dtTheta oracle | 2160 samples, motion error 1.24829e-14 |
| compact boundary oracle | 2160 samples, metric error 4.56474e-14 |
| all-61 RHS oracle | 4320 samples, error 2.84217e-14, compatible and standard Phi |
| exact Minkowski cycle zero | max error 0 |

The bounded one-/eight-tile Aurora PVC rerun also passed.  Aurora debug job
`8790864` completed on `x4216c4s4b0n0` with eight ranks mapped to eight distinct
PVC tiles.  The evolved dynamic-q one-/eight-rank conditioned history
difference was `3.88980825583101983e-14`, below the unchanged `5e-12`
tolerance.  This is a short portability/equivalence gate, not puncture
stability evidence.  Compact evidence is in:

- `artifacts/ref_gh_ordering_gauge_discriminator_20260829/phase0_local/`;
- `artifacts/ref_gh_ordering_gauge_discriminator_20260829/phase0_aurora_8790864/`.

## Diagnostic-only implementation checkpoint

The current branch adds diagnostic support required by the next discriminator
phase without intentionally changing the evolution equations:

- globally associates the initial-RHS maximum with the MPI rank, component,
  and radius that produced it;
- records RHS-family maxima and locations for Psi, Pi, Phi, Hhat, theta, and
  Upsilon;
- records the physical `sqrt(gamma_ij beta^i beta^j)/alpha` globally and in
  fixed radial regions, including the first radius where it reaches one;
- excludes reduction, curl, and RHS diagnostics whose finite-difference
  stencil overlaps the puncture when the existing puncture-stencil diagnostic
  exclusion is enabled;
- optionally decomposes the cycle-zero production RHS into principal,
  covariant-vacuum, ordinary-gauge, gamma0, gamma2, driver, and KO sectors,
  then requires their sum and an exact production rerun to agree.

Fresh post-instrumentation local checks passed: the complete source-unit gate
retained its original tolerances and all oracle maxima, a 16^3 cycle-zero
sector smoke reproduced the production RHS with conditioned error
`3.94430452610505903e-31` and exact rerun difference zero, and cycle-zero
smokes for all four A-D inputs completed.  The full 96^3 MPI/PVC fixed-point
decomposition and evolved A-D matrix have not run.  The diagnostics therefore
remain unqualified on Aurora at this checkpoint.

Two implementation details require explicit review before that run.  First,
the compact analytic backend deliberately reports zero for the legacy
recursive `reference_Riemann`, `spin`, and `spin_derivative` max-location
entries instead of rebuilding oracle-only tensors during ordinary history
diagnostics; those columns are unavailable, not evidence that the underlying
quantities vanish.  Second, each max-location family currently launches a
separate reduction.  This is acceptable only if the required `dt=0.05`
telemetry remains operationally bounded on PVC; no performance qualification
has been made.

## Remaining controlled work

## Phase 2 full-resolution fixed point

Aurora debug job `8790897` built exact commit `68bc1ee30c0ac64c3afe8d3961bead65efce205a`
on node `x4610c7s7b0n0` and mapped 12 ranks to 12 distinct PVC tiles.  The
96^3 grid contained 216 MeshBlocks, exactly 18 per rank.  The diagnostic
sector sum reproduced the production RHS with conditioned error
`8.07793566946316089e-28`, and the exact production rerun differed by zero.

The globally associated maximum is `5.84252885406934375e-11` in theta
component 54 at `r=0.148779758927976M`, rank 3, gid 63.  Thus the previously
reported radii `2.49052M` and `1.42324M` were rank-local metadata and are
superseded.  At the corrected maximum, the actual RHS is the sum of a driver
contribution `-6.2168543626826e-13` and KO contribution
`5.90469739769617e-11`; all Einstein-sector entries are zero because this is a
theta component.  This identifies the cycle-zero seed but does not establish
which term drives the later exponential mode.

## Phase 3 A-D matrix through 3M

All four cases used the frozen 96^3, `[-2M,2M]^3`, h=M/24, FD4, RK4,
CFL 0.05, KO 0.02, gamma0=1 configuration and the same 12-rank/18-block
decomposition.  Each independent snapshot trajectory produced binary64 state,
native-constraint, and common-ADM files at t=0 and approximately 0.2M.

| Case | Ordering / gamma2 / gauge | Result | Last accepted history |
|---|---|---|---:|
| A | compatible / 0 / off | passed requested gate | 3.000000M |
| B | compatible / 1 / off | passed requested gate | 3.000000M |
| C | compatible / 0 / on | invalid effective timestep at cycle 330 | 1.103057M |
| D | standard / 1 / on | invalid effective timestep at cycle 330 | 1.103057M |

The fatal logs for C and D report the failure at `t=1.123484M`, matching the
earlier full-compatible failure.  Fits over accepted histories with t>=0.2M
give GH-RMS e-folding times `0.037543M` (C) and `0.037517M` (D), and
source-frame-correction e-folding times `0.032512M` and `0.032492M`.
Hhat, theta, and Pi RHS maxima share approximately `0.0376M` growth and end at
`r=0.1487798M`.  A and B instead finish with GH RMS `8.39e-14` and
`2.60e-13`, respectively, and physical metric errors about `3.4e-11`.

The physical chi_beta profile is effectively stationary in A and B despite a
global/inner maximum `38.8265`, a `0.5<=r<1` maximum `1.91125`, and values
below one for `r>=1`.  Therefore `chi_beta>1` alone is not sufficient to
trigger the observed failure: compatible/gamma2-off A remains healthy there.

The predetermined interpretation is now that gamma2 alone and compatible Phi
alone do not explain the regression, while enabling the evolved gauge system
does.  Standard ordering does not repair it.  The primary suspect is therefore
the hyperbolic gauge driver / Einstein-gauge coupling.  The final Phase-8 claim
remains withheld until standard/gamma2 gauge-off Case E and the stationary
driver audit are complete.

Job `8790895` was a pre-build bootstrap failure with no numerical work and no
captured failing command.  The source-external bootstrap logging correction
was commit `68bc1ee3`; rerun `8790897` passed and exited zero.  Both jobs are
preserved as campaign evidence.

## Remaining controlled work

1. Extend healthy A and B from their 3M checkpoints to 5M.
2. Run prescribed Phase-6 Case E (standard, gamma2=1, gauge system off) to 3M.
3. Audit and add the discrete stationary-driver fixed-point oracle without
   changing the target or equations.
4. Derive and evaluate the four requested principal symbols.
5. Make the bounded Phase-8 classification and preserve all compact evidence.

No q feedback, p control, wormhole collapse, moving-center, AMR, binary, or
performance-optimization work is in scope.
