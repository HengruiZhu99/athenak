# Finite-resolution-relaxed Ref-GH q controller

Date: 2026-08-26/27
Branch: `codex/ref-gh-q-relaxed-controller-20260826`
Exact base: `70e1579c2ed117e538e79b4cdf9b461ec58330e8`
Implementation commit: `e403baf03775a04d14c7fa8d50b31a0d44c50c24`
PVC boundary-layout correction: `a3d9818220a509d9b749373f315a2e82ddb44902`
PVC q-reduction staging: `f184fcde3d53b0bed36a855c2a2e07c515abf594`
PVC cache instrumentation: `52dcc573a918b8a794f4273f77b31b7692320edf`
Bounded provider discriminator: `bd40d98b64c4a124e5b8c36679c71a62d1dc6071`

## Status and claim boundary

The analytic, initialization, static-reference, and scalar-controller
foundations are implemented.  A controller-off static reference ladder reaches
`t=0.1M` at `h=M/16,M/24,M/32`.  In the fixed regular annulus
`0.25M <= r < 0.375M`, the `q=0.9` and `q=1.1` representations converge toward
the same stationary physical trumpet at approximately third-to-fourth order.
The exact `q=1` representation remains at binary64 roundoff.

The required prescribed moving-reference and closed-loop qualifications are
**not complete**. Four Aurora PVC attempts passed the one-tile analytic source
unit but suffered an eight-rank Level Zero `NotPresent` GPU write fault in the
evolved path. Equation-preserving boundary and q-reduction refactors narrowed
the latest failure to the provider-profile kernel in the first dynamic
reference-cache rebuild at RK stage two.
No further Aurora job was launched after that discriminator. Consequently this
report does **not** claim finite-resolution-relaxed
control established, moving-reference convergence, closed-loop relaxation,
restart qualification, portability qualification, or production readiness.

## Scientific definition

At a Cartesian active cell with displacement `X^i=x^i-x_p^i`, the production
physical exponent is

```
q_loc = -(1/6) X^k gamma^{ij} partial_k gamma_ij
      = -(1/6) X^k partial_k ln(det gamma).
```

The physical metric and its coordinate derivative are reconstructed directly
from `Psi`, `Phi`, the current reference frame/coframe, and analytic reference
jets.  The production path performs no spatial finite differencing, ray
interpolation, radial interpolation, radial binning, or interpolation to the
puncture.

On actual Cartesian cells in `2h <= r < 8h`, it forms

```
w_i   = (2h/r_i)^3,
q_est = sum_i(w_i q_loc_i) / sum_i(w_i),
```

and records weighted variance, effective sample size, extrema, and sample
count.  The exact finite-resolution trumpet comparator applies the same mask,
shell, and weights to the provider-derived finite-radius exponent.  Thus the
finite-grid target is `q_T^analytic(h)`, not one.

For the independent relative-representation diagnostic,

```
epsilon_G = -(1/6) X^k (G^-1)^IJ bar_theta^K_k Phi_KIJ,
q_phys_loc = q_ref_loc + epsilon_G.
```

The identity is checked on analytic/random states and `epsilon_G` is retained
as a diagnostic rather than forced to zero.

## Complete puncture-stencil exclusion

Every primary field, constraint, common-ADM, exponent, gauge, and cache-oracle
diagnostic discards a sample if the complete finite-difference support can
overlap the puncture.  The implementation uses a conservative axis-aligned
support box with each block direction's actual spacing.  A centered order
`2p` derivative has radius `p`; enabled matching KO dissipation increases the
excluded radius by one cell.  The common ADM path uses the larger of its own
operator footprint and the Ref-GH evolution footprint.

The focused anisotropic-spacing oracle passes.  For the estimator calibration,
the mask retains 2,112 of 2,144 candidate cells.  No post-hoc mask was added
after examining evolution results.

## Static analytic oracles

- Minkowski gives `q_loc=0` to roundoff.
- The exact wormhole uses
  `q_exact(r)=M/(r+M/2)` and the first-order-state estimator agrees at selected
  Cartesian points to expected binary64 accuracy.
- The stationary trumpet uses the same interpolated profile and analytic
  profile derivative as the provider.  Pointwise and weighted estimates agree
  with the finite-radius oracle; they are not compared pointwise with one.
- The analytic weighted trumpet values for `h=M/16,M/24,M/32` are
  `0.8437263470`, `0.8938418434`, and `0.9202374342`, demonstrating the expected
  trend toward one.
- The same wormhole exercise trends toward two under refinement.

The old direct-FD estimate on a fixed `r/h` shell remains only a diagnostic.
Its nonconvergence is expected for a self-similar singular profile.  The
fixed-coordinate annulus direct-FD check converges and remains diagnostic-only.

## q-controlled reference and reprojection

The new `trumpet_q_controlled` reference starts from the exact stationary
trumpet spatial Cholesky factor `L_T` and changes only its singular scaling:

```
W(r) = exp[-(r/R_G)^2],
L(q) = L_T exp[-(q-1) W(r) ln(r/M)],
R_G  = 3M.
```

The lapse, shift, finite-radius trumpet profiles, and Gaussian width remain
fixed.  At `q=1`, the provider reduces to `L_T` to binary64 accuracy.

For `q != 1`, the physical stationary metric is analytically projected into
the current reference frame.  `Psi`, `Phi`, and the generally nonzero `Pi` are
constructed from the same coordinate metric and derivatives.  Reconstructing
the physical metric gives maximum metric error `4.44e-16` and derivative error
`9.86e-16` in the source unit.  The minimum nontrivial projected `|Pi|` is
`0.125187`, confirming that stationary coordinate data do not imply `Pi=0` in
a mismatched spatial reference.

The physical ordinary-GH source and stationary improved-driver `theta` are
projected into the current tetrad.  With gauge-reference subtraction, the
evolved differences subtract the matching current-reference baseline.
Reconstruction errors are `1.11e-16` for `Hhat`, `2.22e-16` for `theta`, and
`2.22e-16` for the subtraction oracle.

Physical boundaries for `q != 1` project the same exact stationary metric and
gauge state at the current stage `q,q_dot,q_ddot`; they never stuff
`Psi=eta,Pi=Phi=0` for a mismatched reference.

## Controller and RK ordering

The replicated state is `(q,q_dot)` with restart persistence of the state,
generation, frozen/enabled state, and latest estimator diagnostics.  The
closed-loop acceleration is

```
dq/dt       = q_dot,
dq_dot/dt   = -2 zeta omega_q q_dot + omega_q^2 (q_est-q),
```

with a smooth acceleration limiter.  The defaults are `zeta=1`,
`omega_q M=0.5`, `q in [0.5,2.5]`, `|q_dot|M<=0.25`.  Crossing a state bound is
a failed run; the state is never clipped and continued.

The PDE and controller use the same RK stage coefficients and stage times.
Each stage copies the PDE and controller stage state, builds a lightweight
current-q spatial snapshot, measures `q_est`, computes `q_ddot`, builds the
full two-jet cache with that same generation, computes the Ref-GH/gauge RHS,
and performs the common RK update.  Generation assertions reject stale-cache
use.

A manufactured scalar test verifies sign, critical damping, slowly varying
lag, exact restart continuation, the prescribed C2 pulse, and lack of clipping.
The critical-history maximum error is approximately `1.03e-15`; the local
prescribed finite-difference oracle error is `1.14e-8`, and the Aurora/SYCL
source-unit value is `3.02e-9`.

## Controller-off static-reference ladder

All nine OpenMP runs use the same physical domain `[-0.5M,0.5M]^3`, exact
projected outer boundary, CFL `0.05`, RK4, fourth-order finite differences,
`gamma0=gamma2=1`, no KO, and the complete puncture-stencil mask.  Only `q` and
the active cells (`16^3`, `24^3`, `32^3`) vary.  Every case reaches `t=0.1M`
finite with no intervention.

In the fixed regular annulus `0.25M <= r < 0.375M`:

| q | h/M | metric Linf | native constraint Linf | lapse Linf | shift Linf |
|---:|---:|---:|---:|---:|---:|
|0.9|1/16|1.5568e-3|1.5670e-2|6.4446e-7|9.8937e-7|
|0.9|1/24|3.6211e-4|4.1008e-3|1.7590e-7|9.5102e-8|
|0.9|1/32|1.2322e-4|1.4604e-3|3.8504e-8|1.4421e-8|
|1.1|1/16|2.0136e-3|1.5609e-2|7.9275e-7|1.0139e-6|
|1.1|1/24|4.8115e-4|4.0819e-3|2.1548e-7|9.9415e-8|
|1.1|1/32|1.6607e-4|1.4529e-3|6.9710e-8|2.0521e-8|

The observed metric orders are `3.60,3.75` for `q=0.9` and `3.53,3.70` for
`q=1.1`; constraint orders are `3.31,3.59` and `3.31,3.59`, respectively.
The `q=1` physical errors remain between roughly `1e-15` and `1e-12`, so no
order is assigned at the roundoff floor.

The innermost `0.125M <= r < 0.25M` physical-metric Linf is **not monotone**:
`q=0.9` gives `4.8404e-2,6.5666e-3,1.1201e-2`, and `q=1.1` gives
`6.6575e-2,1.2940e-2,1.5590e-2`.  This is retained as a negative result.  The
innermost field and constraint observables improve, but the physical-metric
maximum prevents a blanket convergence claim for that region.

At `t=0.1M`, `q_est-q_T^analytic` remains about `+1.1e-5` to `+1.3e-5` for
static `q=0.9`, and `-2.0e-5` to `-2.4e-5` for static `q=1.1`.  These are
representation-mismatch diagnostics, not controller equilibria.

## Prescribed moving-q evidence

A local full-output `q:1 -> 0.9 -> 1` startup reaches `t=0.05M` finite.
Trajectory errors are about `1e-15` in `q` and `2.4e-14` in `q_dot`; the final
physical metric, lapse, and shift errors are `8.27e-11`, `3.18e-13`, and
`1.24e-12`, and the native constraint Linf is `3.47e-9`.  This is only an early
moving-reference smoke test.

Four local full-pulse attempts (`0.9`, `1.1`, `0.75`, `1.25`) were stopped by
the agent at `t=0.284454M` after they proved too expensive for local OpenMP.
Their partial histories are retained, but they do not establish a completed
excursion or convergence.

## Aurora PVC evidence and blocker

Campaign root:

```
/lus/flare/projects/CompactBinaryMerger/hzhu/
  refgh_q_relaxed_20260826_e403baf_v1
```

Job `8785612` was a wrapper-only failure: PBS placed its stdout file inside the
fresh checkout, causing the strict clean-tree assertion to exit before build.
The zero-byte stdout and complete `qstat` record are retained.

Job `8785680` used implementation commit `e403baf0`.  It demonstrated twelve
distinct PVC tile mappings, configured Kokkos `SERIAL;SYCL` with
`Kokkos_ARCH_INTEL_PVC=ON`, built successfully, and passed the one-tile
source-unit/cache gate.  The first eight-rank `q=0.9` prescribed case completed
setup and wrote `t=0` histories, then all ranks reported a Level Zero GPU write
fault (`NotPresent`, PDP level) during the first evolved cycle.  Exit status was
134.

The first hypothesis was excessive private state in the exact q-controlled
boundary kernels: the PVC compiler reported roughly 119 spilled registers.
Commit `a3d98182` split metric and gauge projection and changed the launch to
one ghost cell per work item, exactly preserving local numerical histories.
The new kernels spill only about 19 and 17 registers.

Job `8785718` rebuilt `a3d98182`, passed the same one-tile source-unit/cache
gate, and reproduced the same eight-rank first-cycle `NotPresent` write fault
on a different node.  Therefore boundary spill pressure was reduced but is not
the established root cause.

Commit `f184fcde` then staged the q estimator into one current-q sample kernel
and separate sum and min/max reductions, preserving the estimator and
controller mathematics. A local fenced full-output cycle matched all
pre-refactor numerical histories exactly. Aurora job `8785796`, charged to
`CompactBinaryMerger` in the `debug` queue, rebuilt that commit, demonstrated
12 distinct rank-to-tile mappings, and passed the one-tile source-unit/cache
gate. The evolved case used eight ranks. All ranks completed the first RK stage
and the next stage's `CopyU` and q sample/sum/extrema tasks. No
rank reached the next `UpdateReference` fence before the Level Zero write
fault. This rules out the former combined q reduction as the established root
cause and localizes the blocker to the first dynamic `FillReferenceCache`
rebuild at RK stage two. The exact cache subkernel remains unknown. Static-q
source-unit cases explicitly disable `reference_time_dependent`, so their pass
does not qualify this dynamic path.

Commit `52dcc573` added fence-only labels between every production cache
subkernel. Commit `bd40d98b` added a submission-time `CYCLE_GATE_ONLY=1` stop
that prevents a successful discriminator from falling through into expensive
pulses. Aurora job `8785833` used that bound in the `debug` queue. Initial cache
construction passed every labeled subkernel on all eight evolved ranks, and
the first RK stage completed. At the next stage, all ranks passed the q
estimator but none reached the provider-profile fence. The Level Zero write
fault therefore occurs inside the q-controlled provider-profile launch on its
first dynamic rebuild; no downstream frame, connection, mixed-gauge, theta,
spin, or curvature kernel starts. The present provider kernel constructs three
33-Real jets per work item, making private/spill state a portability hypothesis,
not an established conclusion.

No owned Aurora job remained queued or running after `8785833`.

## Phase audit

| Phase | Evidence status |
|---:|:---|
|0|Pass: invalid same-r/h direct-FD gate is diagnostic-only.|
|1-5|Pass locally/source-unit: production estimator, finite-radius oracles, weighted shell, finite-h target, epsilon_G identity.|
|6-9|Pass analytically/local: provider, metric/gauge reprojection, exact current-q boundary algebra. PVC evolved boundary/runtime remains unqualified.|
|10-12|Implemented and locally tested: state, ODE, same-RK ordering, generation guards, restart serialization code. Runtime restart gate pending.|
|13|Partial: analytic jets and early local smoke pass; complete four pulses and moving-reference convergence missing; PVC q-provider rebuild fails.|
|14|Pass manufactured scalar histories; per-resolution equilibrium runtime test pending.|
|15|Partial: all static cases finite through 0.1M and regular annuli converge; innermost metric Linf nonmonotone.|
|16-18|Not run: closed-loop, large mismatch, and three-resolution finite-h equilibria.|
|19|Pass as diagnostic policy; fixed-r/h nonconvergence retained.|
|20|Honored: all work is uniform-grid; no SMR qualification attempted.|
|21|Not pass: PVC analytic gate passes but evolved cycle fails; runtime restart and full host/MPI/device matrix remain incomplete.|

## Artifacts and reproduction

Compact evidence is under
`docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/`.  It includes the
nine static histories and logs, generated convergence JSON/Markdown, local unit
logs, interrupted local pulse histories, four Aurora failure bundles, exact
rank-to-tile mappings, compiler/configuration provenance, qstat records, and
compact SHA-256 manifests.  Large field output and the 16 MB Aurora restart are
not committed.

Regenerate the static table with:

```bash
python3 scripts/ref_gh/analyze_q_relaxed_controller.py \
  --static-root docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/local/static_t0p1 \
  --json docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/static_t0p1_convergence.json \
  --markdown docs/fo_gh_artifacts/ref_gh_q_relaxed_controller_20260826/static_t0p1_convergence.md
```

## Remaining gates

Before any success claim, the first-cycle PVC fault must be localized and
fixed with a focused full-output evolved gate.  Then the four complete
prescribed pulses, prescribed moving-reference resolution ladder, closed-loop
`q(0)=0.9,1.0,1.1` ladders, untuned `0.75/1.25` cases, finite-h equilibria,
physical/constraint convergence, exact restart trajectory, host/MPI/CUDA/PVC
matrix, and no-intervention audit must all pass.  Until then the controlling
success phrase is explicitly withheld.
