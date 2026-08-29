# Ref-GH ordering, gamma2, and hyperbolic-gauge discriminator

Date: 2026-08-29 (America/New_York)

## Claim status

`GAUGE-DRIVER COUPLING DEFECT ISOLATED`

The standard Einstein/reference system remains finite and roundoff-small with
the evolved gauge sector disabled, while both compatible and standard systems
develop the same fast inner mode when that sector is enabled. The failure is
therefore not repaired by standard Phi ordering and is not caused by gamma2.

This is an isolation result, not a repair or stability qualification. No
gauge-enabled matched trumpet is established beyond `t=1.103057M`; no q
feedback, p control, wormhole, AMR, moving-puncture, binary, or performance
claim follows.

## Exact checkpoint and historical control

- Parent branch: `codex/ref-gh-single-puncture-robustness-20260829`
- Parent commit: `39069a16d2d36e1bf5d124f7f274382eca4cd441`
- Discriminator branch: `codex/ref-gh-ordering-gauge-discriminator-20260829`
- Frozen production source at branch creation:
  `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`
- Starting `git diff --exit-code a09caf70 -- src`: pass

The historical `t=20M` campaign evolved 50 Einstein fields, compatible Phi,
gamma0=1, gamma2 absent/effectively zero, no evolved Hhat/theta/Upsilon, and
KO=0.02. Its exact stationary initial RHS was `O(1e-16)`. The failed current
case evolves 61 fields, compatible Phi, gamma0=gamma2=1, the improved gauge
driver, reference-gauge subtraction, and KO=0.02. These are not identical
systems.

## Frozen regression and portability gates

All Phase-0 gates passed without tolerance changes:

| Gate | Result |
|---|---:|
| deterministic SymPy regeneration | byte-identical |
| analytic coefficient oracle | 216 samples, max error `8.88178e-15` |
| expanded radial oracle | 2160 samples, conditioned error `1.48837e-13` |
| generated geometry oracle | 2376 samples, conditioned error `2.33147e-15` |
| moving gauge/dtTheta oracle | 2160 samples, error `1.24829e-14` |
| compact boundary oracle | 2160 samples, metric error `4.56474e-14` |
| all-61 RHS oracle | 4320 samples, error `2.84217e-14`, both Phi orderings |
| exact Minkowski cycle zero | max error zero |

Aurora job `8790864` passed the bounded one-/eight-tile dynamic-q cycle with
eight distinct PVC mappings. The conditioned history difference was
`3.88980825583101983e-14`, below `5e-12`. This is portability/equivalence
evidence only.

## Diagnostic implementation

Diagnostic-only source changes on this branch:

- globally associate maxima with the producing MPI rank, component, cell,
  coordinates, and radius;
- record maxima and locations for all six RHS families;
- record physical `chi_beta=sqrt(gamma_ij beta^i beta^j)/alpha` by fixed region;
- discard diagnostic samples whose complete FD/KO stencil overlaps the
  puncture;
- decompose the exact cycle-zero RHS into principal, covariant vacuum,
  ordinary gauge, gamma0, gamma2, driver, and KO sectors and require exact
  reproduction of production arithmetic.

The analytic backend intentionally leaves legacy recursive spin/Riemann
max-location columns unavailable rather than rebuilding oracle-only tensors.
No evolution equation was changed by this discriminator branch.

## Cycle-zero decomposition

Aurora job `8790897` used 12 ranks on 12 distinct PVC tiles, a 96^3 grid, 216
MeshBlocks, and 18 blocks per rank. The sector sum reproduced production with
conditioned error `8.07793566946316089e-28`; an immediate production rerun was
identical.

The corrected global maximum is `5.84252885406934375e-11` in theta component
54 at `r=0.148779758927976M`. The old radii `2.49052M` and `1.42324M` were
rank-local metadata and are superseded. At the true maximum:

| contribution | value |
|---|---:|
| gauge driver | `-6.21685436268260e-13` |
| KO | `5.90469739769617e-11` |
| total | `5.84252885406934e-11` |

This identifies the seed, not the cause of its later amplification.

## Decisive evolution matrix

Every case used the same exact unit-mass q=1 stationary trumpet, 96^3 grid on
`[-2M,2M]^3`, h=M/24, 16^3 MeshBlocks, FD4, RK4, CFL 0.05, KO 0.02, gamma0=1,
physical boundaries, and 12-rank decomposition.

| Case | Ordering / gamma2 / gauge | Result |
|---|---|---:|
| A | compatible / 0 / off | finite through fresh `t=5M` |
| B | compatible / 1 / off | finite through fresh `t=5M` |
| C | compatible / 0 / on | failed at cycle 330, `t=1.123484M`; last history `1.103057M` |
| D | standard / 1 / on | same failure cycle/time and growth rate as C |
| E | standard / 1 / off | finite through fresh `t=3M` |

Fresh A/B `t=5M` controls from Aurora job `8790947` ended with:

| Case | GH RMS | reduction RMS | curl RMS | metric-error RMS |
|---|---:|---:|---:|---:|
| A | `7.3541e-14` | `7.7012e-12` | `8.2445e-14` | `5.7298e-11` |
| B | `3.2376e-13` | `2.0703e-12` | `2.7643e-13` | `5.6837e-11` |

Both have `bad_state=0`, regular-state maximum within `1.4e-12` of one, and
no fast mode. The earlier A/B checkpoint continuations in job `8790932`
failed before an accepted step because the restart launched with an invalid
effective timestep. They are preserved restart failures and are not used as
evolution evidence; the fresh runs supersede them.

C and D have GH-RMS e-folding times `0.037543M` and `0.037517M` and
source-frame e-folding times `0.032512M` and `0.032492M`. Their Hhat, theta,
and Pi RHS maxima grow together at `r=0.1487798M`. Standard ordering therefore
does not change the observed mode.

The maximum characteristic speed remains `0.611936`. A signal from a face at
2M to the failing radius cannot arrive earlier than approximately
`(2-0.14878)/0.611936=3.03M`, later than the failure. The physical boundary is
not a causal explanation for this trajectory.

## Stationary-driver fixed-point oracle

Job `8790947` evaluated the exact cycle-zero q=1 state on three grids. Each
sector decomposition passed the unchanged `5e-13` conditioned reproduction
gate and the repeated production RHS was identical.

| N^3 | h | first included r | `|F-Href|` | driver Hhat | driver theta | actual theta | KO theta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64^3 | 1/16 | 0.223170 | `3.04e-13` | `4.50e-13` | `2.69e-13` | `2.81e-12` | `2.77e-12` |
| 96^3 | 1/24 | 0.148780 | `2.83e-12` | `3.36e-12` | `3.00e-12` | `5.84e-11` | `5.90e-11` |
| 128^3 | 1/32 | 0.111585 | `1.50e-11` | `1.69e-11` | `1.11e-11` | `4.29e-10` | `4.20e-10` |

Upsilon residuals remain `1.37e-15`, `2.28e-15`, and `3.01e-15`. The first
included point stays at approximately fixed `r/h`; it moves into the singular
trumpet as h decreases. Consequently these global maxima are not a
fixed-coordinate truncation sequence. Fits against the moving minimum radius
give approximately `r^-5.2` for stored/driver Hhat, `r^-5.4` for driver theta,
and `r^-7.27` for actual/KO theta. This quantitatively identifies binary64
cancellation in independently reconstructed singular baselines, followed by
KO amplification of the spatially varying residual.

The continuous stationary residual should vanish. The current regularized
storage therefore does not produce a uniform roundoff fixed point as the
puncture is approached. This is a concrete numerical coupling defect and a
credible seed for the observed mode. The data do not yet distinguish whether
the subsequent exponential amplification is solely this nonuniform
cancellation or also a lower-order instability of the chosen puncture gauge
target.

## Formulation and code review

The review used the production code and Lindblom--Szilagyi, arXiv:0904.4873,
Eqs. (9), (11), and Appendix B.

- `ComputeGaugeDriverRhs` implements
  `dt H-beta^i d_i H=-mu(H-F)+theta` and
  `dt theta+eta beta^i d_i H=-eta theta` with the correct signs. Its
  moving-frame Omega terms agree with direct coordinate/frame conversion.
- Exact stationary Hhat and theta initialization is mathematically correct:
  theta cancels shift advection and frame motion. For q=1, however, stored
  deltas are obtained by subtracting two independently evaluated singular
  expressions. They are only approximately, not bitwise, zero.
- Reference subtraction reconstructs raw Hhat, theta, and spatial derivatives
  consistently. Static q=1 correctly has no `dtTheta` subtraction;
  time-dependent `dtTheta` is separately oracle-qualified.
- The Einstein source uses the same-stage gauge RHS plus the analytic baseline
  time derivative for `d_t Hhat`. The all-61 oracle includes this path and
  passes in both Phi orderings.
- The physical 1+log/non-advective Gamma-driver target agrees with the
  constraint-satisfying stationary source to the residuals above.
  `tildeGamma` and Upsilon RHS remain at roundoff, so Upsilon initialization is
  not the seed.
- Physical boundary gauge data use the same stationary projection as initial
  data. The location and causal-time calculation independently exclude the
  boundary from the observed failure.

No algebraic sign, frame-motion, `dtTheta`, same-stage derivative, or boundary
projection error was found. The next equation-preserving repair candidate is
to evaluate the regular differences `F-Fref`, Hhat derivatives, and gauge
source increment directly in compact residual form and to make the exactly
matched q=1 stored deltas bitwise zero. It must retain the generic path as an
oracle and pass all-61 equivalence before an evolved rerun. That repair is not
implemented or claimed here.

## Principal symbols

For one symmetric metric component and a physical unit covector `s_I`, define
`beta_s=beta^I s_I`. With the code's gamma1=-1 convention, the standard
Einstein principal symbol is

```text
A Psi      = 0
A Pi       = gamma2 beta_s Psi - beta_s Pi + alpha s^I Phi_I
A Phi_I    = -alpha gamma2 s_I Psi + alpha s_I Pi - beta_s Phi_I.
```

Compatible ordering replaces the final term by
`-s_I beta^J Phi_J`. Gamma2 changes characteristic fields but not speeds.
The improved gauge coupling adds

```text
A Pi_AB   += alpha (s_A Hhat_B + s_B Hhat_A)
A Hhat_A   = -beta_s Hhat_A
A theta_A  = eta beta_s Hhat_A
A Upsilon  = 0.
```

Standard speeds are `0`, `-beta_s`, and `-beta_s +/- alpha`; its complete
driver-coupled symbol has 61 eigenvectors. Over ten trumpet radii from 0.03M
to 5M and 37 directions per radius, the maximum imaginary roundoff was
`2.19e-15`, the minimum geometric dimension was 61, and the worst
amplitude/repeated-eigenspace-invariant basis condition was `4.686`.

Compatible ordering has zero-speed transverse Phi fields. Wherever
`chi_beta=|beta|/alpha>=1`, a direction exists with `beta_s=alpha`; one wave
speed then collides with zero while a transverse shift remains. The 50-field
symbol has only 40 independent eigenvectors instead of 50 at that direction.
This occurs at every sampled radius through 0.8M, where chi_beta decreases
from 47.57 to 1.051. Thus compatible ordering is not strongly hyperbolic in
that region and should not be promoted for puncture production.

This mathematical defect is separate from the current failure: gauge-off A
survives through 5M despite entering the defective region, while standard D
fails identically to compatible C. The observed `t=1.123484M` mode is a
lower-order gauge-driver/coupling problem, not the compatible principal-symbol
collision.

## Artifacts and final boundary

Compact evidence is under
`artifacts/ref_gh_ordering_gauge_discriminator_20260829/`, including jobs
`8790864`, `8790895`, `8790897`, `8790932`, and `8790947`, all histories,
max-location tables, sector tables, mappings, provenance, analyses, and
negative results. Large Phase-3 CBIN/restart files remain in the recorded
Aurora campaign directory and were not committed.

No performance profiling was run because the controlling task explicitly
stopped performance optimization until the formulation defect is repaired.
There are no active Aurora jobs for this campaign at handoff.

Final classification: `GAUGE-DRIVER COUPLING DEFECT ISOLATED`.
