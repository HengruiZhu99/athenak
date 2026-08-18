# FO-GH puncture validation status

## 2026-08-17 revised 20M SMR campaign: paused formulation gate

The revised `[-32M,32M]^3`, 232-block, eight-A100 campaign is paused. FO-GH
coarse, medium, and fine runs reached valid 2M restarts but suffered timestep
collapse at `3.431611M`, `3.024995M`, and `2.658676M`. Earlier failure with
finer resolution is a failed stability/convergence gate and makes a formulation
or high-frequency semidiscrete instability a leading hypothesis, not a proven
root cause. With outer faces at 32M and pre-collapse characteristic speed near
0.964, outer-boundary arrival cannot explain failure before 3.5M.

CUDA-aware MPI, eight GPU mappings, exact eight-rank Minkowski, revised-tree
startup, restart, and one/eight-rank comparison passed. Z4c production never
launched, so no comparison or stability claim through 20M exists. Common
unmasked ADM momentum histories are identically zero despite nonzero native
momentum. A subsequent audit confirmed fill-before-initialization defects in
the common ADM Christoffel and covariant-derivative index raising, plus NaN
masking in history output. The operator and two ordering-sensitive manufactured
flat-space regressions are repaired, but all common H/M histories in the paused
bundle are invalid and must be regenerated. This diagnostic-only defect does
not explain the evolution collapse. Compact evidence and the review prompt are in
`docs/fo_gh_artifacts/perlmutter_20m_20260817_partial/`.

Status date: 2026-08-17

This document records evidence, including failures.  It is not a production or
long-stability qualification claim.  The working branch is
`codex/fo-gh-puncture-driver-20260817`, based on
`24dd527514a3b031d151ca8d3f2679e998a91b3d`.

## Pending revised 20M identical-grid comparison

The revised campaign requests three FO-GH and three Z4c runs through `t=20M`
on a fixed `[-32,32]^3` octree, using 32, 48, and 64 active cells per
MeshBlock.  Both controlling inputs reproduce the authoritative 232-block
tree: 56, 56, 56, and 64 leaf blocks on physical levels 0--3.  Their leaf
envelopes are `[-32,32]^3`, `[-16,16]^3`, `[-8,8]^3`, and `[-4,4]^3`, so the
finest level covers the requested `[-2,2]^3` cube.  The finest spacings are
`1/16`, `1/24`, and `1/32 M`.  Compact evidence is in
`docs/fo_gh_artifacts/local_20m_campaign_preflight/README.md`.

The common ADM path uses one fourth-order AthenaK stencil for both
formulations, no evolving lapse/chi mask, fixed radial regions, and fixed
shells around every refinement interface.  Native FO-GH alpha-masked and Z4c
chi-masked histories remain secondary outputs.  Checkpoints are requested
every `2M`, and history/slice output every `0.2M`.  No stability claim beyond
`20M` is permitted.  The previous 3,200-block `[-128,128]^3` run was stopped
before `t=2M` when this revised campaign superseded it.

## Scope used for this campaign

The campaign is vacuum FO-GH only.  Fluid coupling, Kerr-Schild validation, and
apparent-horizon work are intentionally excluded by later user direction.  The
scientific gates retained here are Minkowski stability, uniform and real-SMR
wave behavior, compatible gradients, AMR/restart plumbing, lapse-masked
constraint convergence, and identical-data puncture stability.

## Local verification

The current host lacks an importable `pytest` installation, so the AthenaK test
executables were run directly with the exact test inputs and overrides rather
than claiming a pytest harness pass.  The Release/Serial executable used for
the longer local checks was:

```
/tmp/athenak-fogh-release.m7l1hu/src/athena
```

Observed local results:

- tensor, algebra, geometry, primary RHS, and compatible-gradient device
  problem generators passed;
- the compatible-gradient device problem now uses nonzero shift and driver
  profiles and checks the production scalar/vector/tensor robust-advection RHS
  against a direct `Lx` oracle; it also proves the dataset differs from centered
  advection;
- exact Minkowski remained exact on uniform and real-SMR meshes;
- compatible reduction gradients remained at roundoff;
- after the robust-advection correction, uniform linear-wave errors at
  `N=8,16,32` were `1.1837284e-10`, `7.8606381e-12`, and `5.1512035e-13`,
  giving orders `3.91255` and `3.93166`;
- the real-SMR wave result is only about order `1.5--1.6` and must not be
  described as fourth-order SMR convergence;
- dynamic regrid changed one root block to eight children and repaired all
  compatible gradients with maximum residual near `1.15e-14`;
- direct two-cycle puncture evolution and checkpoint/restart evolution produced
  identical 41-field final checkpoint arrays.

A later independent non-diagonal metric-jet audit now checks conformal Ricci
against the direct coordinate definition of the Ricci tensor and separately
checks `D_i c^i`, Hamiltonian, and momentum.  It passes on the local
Release/Serial backend.  That audit also found and corrected two coupled
non-diagonal-tensor defects in the `Lambda^i` RHS: the `Atilde^{ik} a_k` and
`Atilde^{ik} X_k` terms had used a once-raised mixed tensor, and the symmetric
twice-raised `Atilde^{ij}` workspace had accumulated off-diagonal components
twice.  A non-diagonal RHS regression now covers the corrected contractions,
`Atilde_ij Atilde^ij`, `K`, and `pi` assembly.  A second full nonlinear oracle
covers both `Atilde` trace-free projections, shift second derivatives, explicit
advection, every `Lambda` source, and its vector Lie-index term.  Both focused
unit executables pass from `/tmp/athenak_fogh_geometry_audit/src/athena` with
Kokkos Serial 4.4.0 and GCC 13.3.0.

A subsequent complete non-diagonal ADM-jet oracle found a separate
standard-GH conversion defect: `RegularToStandardGh` contracted the local
`gamma(i,k)` workspace into `d0gamma(i,j)` before all symmetric metric
components had been initialized.  Splitting metric/K reconstruction from the
`d0gamma` pass fixes the nine resulting `Pi_ab` mismatches.  The independent
oracle now checks every regular variable and every reconstructed `g_ab`,
`Phi_iab`, and `Pi_ab` component.  Algebra, geometry, RHS, compatible-gradient,
and tensor device tests pass with the corrected Serial executable, SHA-256
`8de429a9a0983dacd034cc7fa7183a5c9ec68b04b7b9aedfb09d31c3cf1368b5`.

All Perlmutter and `t>=1M` puncture evolutions reported below predate this
continuum correction.  They remain useful failure-localization evidence but do
not qualify the corrected source.  A bounded corrected-source local subset now
passes: exact uniform/SMR Minkowski; robust uniform/SMR ratios `0.95388` and
`0.99244`; compatible gradients; uniform wave orders `3.91255` and `3.93166`;
real-SMR wave order `1.51443`; a finite `t=0.02M` puncture with 22 history
columns and 43 checkpoint fields; and the `t=0.01M` three-resolution ladder
immediately below.  This is not a replacement for corrected-source GPU or
long-puncture qualification.

The corrected-source lapse-masked puncture constraint ladder at `t=0.01M` was:

| N | Hamiltonian L2 | Momentum L2 | GH L2 | Reduction L2 |
|---:|---:|---:|---:|---:|
| 16 | 1.07731e-2 | 8.69365e-5 | 4.92939e-5 | 6.62641e-3 |
| 24 | 4.99979e-3 | 4.98697e-5 | 1.84654e-5 | 5.30022e-3 |
| 32 | 1.99594e-3 | 2.66090e-5 | 7.70410e-6 | 1.39435e-3 |

The `24 -> 32` observed orders are `3.1920`, `2.1835`, `3.0386`, and
`4.6417`, respectively.

The momentum family now follows the Z4c diagnostic convention by contracting
the covariant momentum constraint with the physical inverse metric,
`gamma^{ij} M_i M_j = chi gtilde^{ij} M_i M_j`, rather than using Cartesian
component squares.  The `alpha>=0.25` lapse mask is unchanged.  The checkpoint
and history family L2 values agree to roundoff in the corrected short run.

## Perlmutter provenance

The first isolated campaign used:

```
allocation 57187283
node nid001020
QOS gpu_shared_interactive
partition shared_urgent_gpu_ss11
GPU NVIDIA A100-SXM4-40GB, UUID GPU-78c49639-ef05-0b69-b11c-e26bfd627631
CUDA toolkit 12.9
Kokkos 4.7.2, SERIAL+CUDA, AMPERE80
MPI cray-mpich/9.0.1
compiler source/kokkos/bin/nvcc_wrapper with gcc-native/14
source commit 1e4c62ee25dd443334f8ddab1a27ad4d697e21d7
Kokkos gitlink 6739bc623081648af9e752b616d9671527922cbf
executable SHA-256 11fb6955d027f50e69610e3a6da7b4656c1ff6536d41d1884920da8439acf5a2
```

The remote working directory and retained checkpoints are:

```
/pscratch/sd/h/hzhu/fo-gh-puncture-20260817.L0vwO0
```

`nvidia-smi dmon` observed 1165 MiB framebuffer use, 194 W power, and A100
application clocks while the `N=32`, `t=0.2M` puncture kernel ran.  This is
runtime CUDA evidence rather than compilation-only evidence.

## Perlmutter preflight results

The following one-rank/one-A100 checks completed successfully:

- exact uniform Minkowski: maximum error exactly zero;
- exact real-SMR Minkowski: maximum error exactly zero on nine MeshBlocks;
- robust uniform Minkowski: finite, final/initial perturbation Linf `0.9094`;
- robust real-SMR Minkowski: finite, final/initial perturbation Linf `0.9834`;
- compatible-gradient device test;
- uniform linear wave: orders `3.91255`, `3.93202`;
- real-SMR linear wave: order `1.60701`, explicitly not fourth order;
- dynamic regrid and compatible-gradient repair: `1.15035e-14`;
- identical-data puncture smoke: all 41 checkpoint fields finite;
- puncture direct/restart equivalence: bit-for-bit equal, maximum difference
  zero;
- lapse-masked puncture convergence at `N=16,24,32`, using the then-current
  Cartesian-component momentum magnitude.  That momentum diagnostic is
  superseded by the corrected geometric contraction in the local table above.

A second one-A100 preflight of the later robust-advection/history source
(`3ec9c3bd326f22c7dedf792572876f4c2a8683a1`) also passed: uniform and SMR
robust-Minkowski ratios were `0.909425` and `0.983360`; uniform wave orders
were `3.912554` and `3.932016`; real-SMR wave order was `1.607008`; dynamic
regrid repair left a `1.15035e-14` gradient residual; the puncture history and
checkpoint schemas contained 22 columns and 43 fields; and an actual restart
matched directly evolved data bit for bit.  That source still predates the
non-diagonal `Atilde^{ij}` correction above and therefore is not corrected-RHS
qualification evidence; its momentum histories also use the superseded
component norm.

## Uniform puncture time ladder

The captured Perlmutter campaign used fourth-order centered derivatives,
including the then-current centered explicit shift-advection terms, RK4,
`CFL=0.025`, `kappa=1`, `mu_H=1`, `eta_H=1`, `eta_beta=2`, and `diss=0.02`.
The later `Lx` correction described below means those puncture histories are
diagnostic evidence for the earlier source state, not qualification of the
current robust-advection implementation.
The domain was initially `[-4M,4M]^3`, with one MeshBlock and outflow-labelled
polynomial-extrapolation boundaries.  Evolution used no floors, clipping,
resets, or excision.  Only diagnostic reductions omitted `alpha<0.25` cells.
The historical momentum columns below use Cartesian component squares rather
than the later Z4c-style physical inverse-metric contraction.

At `t=1M` all three resolutions were finite:

| N | Hamiltonian L2 | Momentum L2 | GH L2 | Reduction L2 |
|---:|---:|---:|---:|---:|
| 16 | 9.76865e-3 | 2.02792e-2 | 4.99901e-3 | 6.42595e-3 |
| 24 | 6.07368e-3 | 1.59905e-2 | 4.10551e-3 | 5.09189e-3 |
| 32 | 2.01731e-3 | 7.66091e-3 | 1.61329e-3 | 1.42159e-3 |

At `t=5M` all runs were still finite, but the norms were no longer
resolution-improving:

| N | Hamiltonian L2 | Momentum L2 | GH L2 | Reduction L2 |
|---:|---:|---:|---:|---:|
| 16 | 1.81360e-1 | 1.82485e-1 | 2.30047e-1 | 2.78327e-2 |
| 24 | 1.26654e-1 | 1.88644e-1 | 2.02327e-1 | 2.20226e-2 |
| 32 | 2.58767e-1 | 8.14100e-1 | 3.63534e-1 | 2.32412e-2 |

The first conspicuous masked-norm transitions occur near `2M`, `3.5M`, and
`4.5M` for `N=16,24,32`.  The outer boundary is at coordinate radius `4M` on
each Cartesian face, so a simple flat-space half-box crossing estimate is
`4M`.  That timing initially made boundary contamination plausible.

The completed equal-spacing domain-control ladder rules it out as the cause of
the central behavior through `5M`.  Small-box `[-4M,4M]^3` resolutions
`N=16,24,32` were paired with doubled-box `[-8M,8M]^3` resolutions
`N=32,48,64`.  For every pair, the fixed `r<2M` histories agree to roughly one
percent or better through `5M`; at `t=5M`, the four small/doubled ratios are
`0.9989,1.0007,1.0004,0.9999`, `0.9996,0.9997,0.9997,0.9998`, and
`0.9995,1.0000,1.0000,0.9999`.  Moving the boundary from `4M` to `8M` therefore
does not change the observed central transition on this interval.

The apparent resolution reversals in the whole lapse-masked near region are
also confounded by a changing integration domain.  As cells cross
`alpha=0.25`, the included physical volume jumps at resolution-dependent
times: about `1.75M` (`N=32`), `3.5M` (`N=48`), and `2.5M` plus `4.25M`
(`N=64`).  Thus those masked near norms do not compare the same physical
region at a fixed time.

Subtracting the `r<2M` squared integrals and volume from the global history
gives an exterior `r>=2M` diagnostic.  At `t=5M`, doubled-box
`N=32,48,64` exterior `(H,M,GH,reduction+curl)` L2 values are
`(1.9323e-3,1.0806e-3,1.7264e-3,2.8764e-4)`,
`(6.8110e-4,6.2949e-4,1.7388e-3,1.3048e-4)`, and
`(6.0070e-4,5.1725e-4,1.7516e-3,8.3867e-5)`.  The fine-pair orders are
`0.437`, `0.683`, `-0.026`, and `1.536`.  The exterior does not blow up and
the reduction family improves cleanly, but the GH norm stalls rather than
converging.  This remains a failed convergence gate even before accounting
for the corrected RHS.  Here too, `M` is the superseded component norm; the
other three families are unaffected by the momentum-norm correction.

No `20M`, `50M`, `100M`, or long-SMR claim is valid.  The next numerical step
is to repeat the bounded ladder with the corrected twice-raised `Atilde` RHS
and with diagnostics that distinguish fixed spatial regions from the moving
lapse mask.

## Parameter sensitivity on the small box

A bounded `N=16`, `t=5M` diagnostic sweep varied `mu_H`, `eta_H`, `eta_beta`,
KO strength, CFL, and FD order.  Halving CFL changed the final norms by less
than numerical noise, which disfavors an RK/CFL explanation.  `mu_H=2--4` and
`diss=0.04` reduce some amplitudes but do not establish convergence or justify
changing the production parameters.  `diss=0` remains finite but has larger
momentum and GH norms.  These runs are confounded by the small boundary and are
not promotion evidence.

## Current conclusion and qualification gaps

The pre-correction module passed the narrow algebraic, Minkowski, wave,
compatible-gradient, restart, and regrid checks listed above.  The corrected
non-diagonal RHS and standard-GH map pass focused local tests, a `t=0.2M`
bounded puncture, the `t=0.01M` constraint ladder, and bitwise two-cycle
restart.  They have not been rerun on GPU and have not demonstrated long
puncture stability.  The `5M` loss of resolution improvement is a hard
scientific gate, not a threshold to weaken.  Current evidence does not
establish production readiness, long uniform stability, long SMR stability,
or one-/four-GPU agreement.

The next reviewer should prioritize:

1. independent review of the corrected twice-raised/symmetric `Atilde` logic;
2. a fresh corrected-source preflight and bounded uniform ladder;
3. fixed-radius exterior and near-region histories alongside the moving lapse
   mask;
4. only after those checks, an SMR ladder and continuation to `20M` and beyond.

After the captured campaign, the source was extended to provide the missing
curl, determinant/trace, gauge-residual, and separately normalized fixed-radius
history reductions needed by items 2 and 3.  A later local focused run verified
the 22-column history and 43-field checkpoint schemas through `t=0.02M`; all
values were finite.  The same update routed explicit shift-advection terms
through AthenaK's robust `Lx` operator while preserving the exact compatible
`beta.(Q,X,a,B)` products.  Its direct `Lx` oracle, uniform robust Minkowski,
real-SMR robust Minkowski, primary-RHS unit, and short puncture smoke checks all
pass in the Release/Serial build and were subsequently exercised by the second
one-A100 preflight.  The newer twice-raised/symmetric `Atilde` correction
remains GPU-unverified and does not upgrade the long-stability claim.

The focused post-change commands ran the executable directly from a fresh
temporary directory with `tst/inputs/fo_gh_compatible_unit.athinput`,
`fo_gh_rhs_unit.athinput`, `fo_gh_stability.athinput`, and
`fo_gh_puncture_evolution.athinput`.  The uniform robust-Minkowski final/initial
`Linf` ratio was `0.95388`; the real-SMR ratio was `0.99244`.  Repeating the
wave script's uniform `N=8,16,32` and real-SMR `N=16,32` ladders gave the
uniform results above and SMR order `1.51443`.

All source logs, histories, checkpoint summaries, CMake provenance, executable
hashes, and GPU telemetry used above are under
`docs/fo_gh_artifacts/`.
