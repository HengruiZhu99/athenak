# FO-GH puncture validation status

Status date: 2026-08-17

This document records evidence, including failures.  It is not a production or
long-stability qualification claim.  The working branch is
`codex/fo-gh-puncture-driver-20260817`, based on
`24dd527514a3b031d151ca8d3f2679e998a91b3d`.

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
- exact Minkowski remained exact on uniform and real-SMR meshes;
- compatible reduction gradients remained at roundoff;
- uniform linear-wave errors at `N=8,16,32` were
  `1.1837289e-10`, `7.8606080e-12`, and `5.1499314e-13`, giving orders
  `3.91255` and `3.93202`;
- the real-SMR wave result is only about order `1.5--1.6` and must not be
  described as fourth-order SMR convergence;
- dynamic regrid changed one root block to eight children and repaired all
  compatible gradients with maximum residual near `1.15e-14`;
- direct two-cycle puncture evolution and checkpoint/restart evolution produced
  identical 41-field final checkpoint arrays.

The lapse-masked puncture constraint ladder at `t=0.01M` was:

| N | Hamiltonian L2 | Momentum L2 | GH L2 | Reduction L2 |
|---:|---:|---:|---:|---:|
| 16 | 1.07731e-2 | 2.16305e-4 | 4.92939e-5 | 6.62641e-3 |
| 24 | 4.99979e-3 | 1.65796e-4 | 1.84654e-5 | 5.30022e-3 |
| 32 | 1.99594e-3 | 8.39764e-5 | 7.70410e-6 | 1.39435e-3 |

The `24 -> 32` observed orders are `3.1920`, `2.3645`, `3.0386`, and
`4.6417`, respectively.

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
- lapse-masked puncture convergence at `N=16,24,32`, matching the local values
  above.

## Uniform puncture time ladder

The production test parameters were fourth-order centered differences, RK4,
`CFL=0.025`, `kappa=1`, `mu_H=1`, `eta_H=1`, `eta_beta=2`, and `diss=0.02`.
The domain was initially `[-4M,4M]^3`, with one MeshBlock and outflow-labelled
polynomial-extrapolation boundaries.  Evolution used no floors, clipping,
resets, or excision.  Only diagnostic reductions omitted `alpha<0.25` cells.

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

The first conspicuous growth in the coarse history occurs near `2M`, `3.5M`,
and `4.5M` for `N=16,24,32`.  Because the box width is `8M`, the flat-space
light-crossing time is `8M` and its half-crossing time is `4M`.  The fine-grid
loss near `4.5M` is therefore consistent with outer-boundary contamination.
The initial isotropic physical characteristic speed is
`alpha*sqrt(chi)=alpha^2`; integrating its reciprocal from `r=4M` to `r=1M`
gives about `7.15M`, but the evolved metric/gauge changes that estimate.

A single same-spacing doubled-box control (`[-8M,8M]^3`, `N=32`, equivalent
to the small-box `N=16` spacing) is finite through `5M`.  Its whole-domain L2
norms cannot diagnose central boundary arrival because the global reduction
contains boundary cells immediately.  Its near-puncture final norms closely
match the small-box coarse run.  This one control is insufficient to either
prove or rule out boundary-driven loss of convergence.

The required next experiment is a doubled-box ladder at `N=32,48,64`, which
preserves the central spacings of the original `N=16,24,32` ladder.  Compare
fixed central-region histories before the measured constraint-characteristic
arrival.  No `20M`, `50M`, `100M`, or long-SMR claim is valid until this gate is
resolved.

## Parameter sensitivity on the small box

A bounded `N=16`, `t=5M` diagnostic sweep varied `mu_H`, `eta_H`, `eta_beta`,
KO strength, CFL, and FD order.  Halving CFL changed the final norms by less
than numerical noise, which disfavors an RK/CFL explanation.  `mu_H=2--4` and
`diss=0.04` reduce some amplitudes but do not establish convergence or justify
changing the production parameters.  `diss=0` remains finite but has larger
momentum and GH norms.  These runs are confounded by the small boundary and are
not promotion evidence.

## Current conclusion and qualification gaps

The module has passed the narrow algebraic, Minkowski, wave, compatible-gradient,
restart, and regrid checks listed above.  It has not demonstrated long puncture
stability.  The `5M` loss of resolution improvement is a hard scientific gate,
not a threshold to weaken.  Current evidence does not establish trumpet
behavior, production readiness, long uniform stability, long SMR stability, or
one-/four-GPU agreement.

The next reviewer should prioritize:

1. the correctly scaled doubled-domain characteristic-arrival study;
2. fixed-radius exterior and near-region histories rather than global-only
   normalization;
3. missing curl, determinant/trace, and gauge-residual time-series diagnostics;
4. only after those checks, continuation to `20M` and beyond.

After the captured campaign, the source was extended to provide the missing
curl, determinant/trace, gauge-residual, and separately normalized fixed-radius
history reductions needed by items 2 and 3.  That extension has passed an
incremental Release/Serial compile, but no Athena execution was started after
the user's request to terminate runs.  Its runtime and GPU behavior therefore
remain explicitly unverified.

All source logs, histories, checkpoint summaries, CMake provenance, executable
hashes, and GPU telemetry used above are under
`docs/fo_gh_artifacts/`.
