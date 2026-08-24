# Ref-GH feedback continuation, 2026-08-23

## Status

This branch implements the equation-preserving feedback clock requested in the
controlling goal.  Local and twelve-tile Aurora T0--T2 gates pass, including
exact prescribed-law equivalence and restart continuity.  The required Aurora
tau-8 replay completed through t=4M, and the controller thresholds below were
frozen before any closed-loop run.  T3 has not started, so no closed-loop
continuation or stability claim is made.

The work is based exactly on
`9c438dc619aa742404530c953243d71b2a01d8e6` from
`codex/ref-gh-transition-path-ordering-diagnosis-20260822`.  The implementation
branch is `codex/ref-gh-feedback-continuation-20260823`; the controller code and
inputs are commit `9b86a5d97af9abb46b18c82c4895feb9887f0d5d`, with the
stronger manufactured-history audit in
`55eed5d4267d2eea833c4fe3ec6e8ff3af3bc732`.

Aurora debug job `8777274` passed the complete twelve-tile gate on node
`x4600c0s5b0n0`.  It proved distinct PVC mappings `0.0` through `5.1`, exact
prescribed-law payload equivalence (`Linf=0`), all manufactured controller
histories, and an actual checkpoint/restart.  Restart differences were
`1.1679398845623745e-09` in the spacetime payload,
`1.0869126779167182e-16` in xi, and `2.570860191397628e-15` in xi_dot, all below
their gates.  Earlier jobs `8777229` and `8777250` remain preserved as explicit
mesh-teardown and Python-3.6 harness failures rather than solver evidence.

Capacity job `8777287` replayed the old-medium fixed-core tau-8 path on all
twelve PVC tiles, restarted at t=2M, and completed t=4M.  Its safe envelope was
`kappa_G=3.744620538609028`, `a_max=1.574613081656999`,
`v2_max=0.03321661761817533`, and normalized risk
`0.6349398434951364`.  The accumulated characteristic distance was
`3.3145760789006498M` on the deliberately old `[-6M,6M]^3` calibration domain.
Native normalized maxima were GH `3.064056190379227e-3`, reduction
`8.455093154513271e-4`, and curl `1.4093099168983025e-2`.

The replay exposed a diagnostic-only bug: prescribed modes returned before
copying those native norms into the controller history, leaving its three
columns zero even though the native history was valid.  Commit
`8b7c0841585004c9168d57700f1128f9bbdbca6e` moves that return after norm
publication while keeping all veto state changes feedback-only.  A local
prescribed one-cycle test matches all three independently normalized native
values to `2.65e-23` Linf or better.  No GH equation, controller equation, or
threshold changed.

The first T3 attempt, capacity job `8777374`, passed that focused PVC history
check and completed the feedback segment through t=0.5M.  It remained finite
with `xi=0.04598493014744905`, maximum risk `0.004586673499545036`, no veto,
and native/controller constraint histories agreeing to `3.39e-21` Linf or
better.  The next segment never evolved: all ranks failed to open a relative
`rst/...` path after the launcher changed directories.  PBS consequently
reported exit 143.  This is classified as a launcher failure, not a controller
or solver failure; the script now resolves an absolute checkpoint path before
the focused retry.

The absolute-path retry, job `8777396`, exposed a second orchestration error.
Every 0.5M evolution segment exited zero and the fields remained finite through
t=5M, but the restart command supplied both `-r checkpoint` and the original
input.  This reapplied `continuation_xi=0`: xi returned to
`0.0459849301474` at every segment endpoint instead of accumulating, while the
controller generation advanced.  Its combined trajectory is therefore
scientifically invalid and does not count as T3.  The largest recorded risk
was 0.6513, still below the frozen slow threshold, and there was no constraint
veto.  The launcher now uses the already qualified `-r checkpoint`-only
pattern so restart metadata remains authoritative; no threshold or equation
was changed.

The clean checkpoint-only retry, capacity job `8777507`, passed T3 on node
`x4301c1s7b0n0`.  It reached `xi=0.5008422433746882` at t=2.5M with monotone
xi, nonnegative xi_dot, finite fields, exact zero exponent trims, and no veto.
The controller did not yet slow, which is the permitted T3 exception because
every risk channel stayed strictly below `R_slow=0.70`; the maximum risk was
0.3821703582169683.  Native maxima were GH `1.5705e-3`, reduction
`2.3654e-4`, and curl `5.4938e-3`, and their controller-history copies agreed
to `8.68e-19` Linf or better.  This is a bounded plumbing/discrimination pass,
not a long-time or full-activation result.

## Fixed mathematical scope

The radial family, GH equations, compatible Phi ordering, dissipation, and
profiles are unchanged.  The fixed core is

```
r_core = 0.30 M
kappa_core = 1
B = 0 for r <= 0.30 M
B = 1 for r >= 0.60 M
delta_q = delta_p = 0 exactly
```

Only the prescribed activation clock is replaced.  With the existing quintic
smoothstep `S`, the reference uses `s=S(xi)` and exact constant jets outside
`0<xi<1`:

```
d xi/dt     = xi_dot
d xi_dot/dt = (v_cmd - xi_dot)/tau_v
v_cmd       = v_max F_risk(R) F_end(xi)
R           = max(R_G, R_a_min, R_a_max, R_v)
R_G         = ln(kappa_G)/ln(kappa_stop)
R_a_min     = ln(1/a_min)/ln(1/a_min_stop)
R_a_max     = ln(a_max)/ln(a_max_stop)
R_v         = sqrt(v2_max/v2_stop)
```

Harmless-baseline contributions are zero.  Both governor factors are C2
quintic ramps.  Defaults are `v_max=0.25/M`, `tau_v=0.5M`, and endpoint slowing
over `0.90<xi<1`.  At completion, `xi=1`, `xi_dot=0`, and the activation jet is
exactly stationary.  Reverse motion is prohibited.

The controller state is advanced by the existing low-storage RK recurrence in
the required order: copy PDE/controller base, measure the current relative
state, form the continuation acceleration, rebuild the current reference,
evaluate the GH RHS, and update PDE/controller state.  `xi`, `xi_dot`, cache
generation, veto state, freeze state, completion state, and veto timing are
persisted in restart metadata.

## Frozen safety policy and thresholds

The hard absolute caps are enforced in the parser:

| Quantity | frozen stop | immutable cap |
|---|---:|---:|
| `kappa_G` | 8 | <= 8 |
| `a_min` | 0.5 | >= 0.5 |
| `a_max` | 3 | <= 3 |
| `v2_max` | 0.20 | <= 0.20 |

The risk ramp slows at `R=0.70` and stops at `R=1`.  Job `8777287` established a
maximum safe tau-8 risk of `0.6349398434951364`, so `R_slow=0.70` lies modestly
above the successful envelope.  The absolute stop values in the table and the
risk thresholds are now frozen and must not be retuned after a closed-loop
outcome.

The frozen native-constraint warning levels are GH L2 `2e-2`, reduction L2
`5e-3`, and curl L2 `8e-2`.  The tau-8 maxima are lower by factors of about
6.53, 5.91, and 5.68 respectively.  A warning commands
`v_cmd=0` while evolution continues at fixed commanded xi.  Twice a warning,
or growth after a 0.5M freeze, fails closed.  These are safety vetoes, not
convergence criteria.

## Local evidence

All evidence here was produced by the Release Kokkos Serial build
`build-feedback-local/src/athena`, SHA-256
`e23c7449ba5691c2142b010623d0d9a76ca0589ca45875d94a1c4e08f412cb6c`.

- T0 source/reference tests pass: compatible Phi and source-oracle residuals
  are at or below `6.94e-16`; stationary-trumpet RHS, field, and constraint
  residuals are `7.48e-17`, `1.56e-18`, and `3.12e-18`; dynamic lapse and
  spatial-reference errors are `1.67e-15`.
- T1 legacy-time versus prescribed-xi one-cycle payloads are bitwise equal:
  `Linf=0`, identical cycle and time.
- T2 manufactured safe, approach-stop, evolved excursion/recovery, evolved
  permanently-unsafe, and endpoint histories pass.  The test directly checks
  the activation two-jet chain rule, C2 endpoint limits, exact constant endpoint
  jets, monotone xi, smooth nonnegative rate relaxation, freeze, and resumption.
- Post-schema feedback smoke remains finite for one cycle.  `xi` is monotone,
  `xi_dot` is nonnegative, all four risk channels are finite, and
  `delta_q=delta_p=0` exactly.
- Continuous versus checkpoint/restart comparison passes.  The spacetime
  payload difference is `4.919683421929222e-08` (limit `1e-7`), while
  `|Delta xi|=1.4123e-13` and `|Delta xi_dot|=1.3966e-12` (limit `1e-11`).
  Runtime-written Real metadata now uses `max_digits10`; this representation
  fix removes six-digit restart truncation without changing the equations.

The enlarged `[-12M,12M]^3` mesh audit gives a 3x3x3 root, 272 MeshBlocks in
total, 208 blocks at physical level 1, and 64 at physical level 2.  The finest
logical coverage is `[-4M,4M]^3`, which contains the complete 0.30--0.60M
transition shell.  Medium resolution has `dx_min=M/24`; the planned coarse and
fine variants preserve the tree with `dx_min=M/16` and `M/32`.

T3 accumulated `2.071034491084857M` of maximum-characteristic travel by
t=2.5M.  A linear extrapolation gives `16.5683M` by t=20M, so the 12M faces
would not support a clean time-limit outcome if the controller froze.  Under
the user's explicit SMR-boundary authorization, the definitive input therefore
moves the faces to 24M and adds one coarser outer level without changing the
inner spacing.  Its audited 328-block tree has 208 leaves over `[-24,24]^3` at
`dx=M/6`, 56 over `[-8,8]^3` at `dx=M/12`, and 64 over `[-4,4]^3` at
`dx=M/24`.  Thus the fixed-core shell remains on the finest level while the
projected t=20 characteristic distance retains about 7.43M of margin.  The
projection motivates the grid; only the eventual measured T4 integral will be
used for a causal claim.

## Remaining ordered gates

1. Run T4 on the enlarged medium domain through xi=1 plus a 2M hold, t=20M, or
   fail closed; then T5 aggressive prescribed four-M discriminator.
2. Run the three-resolution T6 gate only if T4 passes.  If the measured
   characteristic travel distance makes the `[-12M,12M]^3` clean window too
   short, push only the outer boundary outward with SMR while preserving the
   fixed inner spacings and full finest coverage of the 0.30--0.60M shell.

No statement of feedback success, convergence, full activation, trumpet
establishment, or long-time stability is justified by the current evidence.

## Reproduction

Local build and mesh audit:

```bash
cmake -S . -B build-feedback-local -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=OFF -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF -DPROBLEM=built_in_pgens
cmake --build build-feedback-local -j
build-feedback-local/src/athena -m \
  -i inputs/ref_gh/ref_gh_feedback_continuation_causal.athinput
```

Aurora submission after creating a fresh campaign clone at the committed SHA:

```bash
qsub -v CAMPAIGN_ROOT=/absolute/fresh/campaign,EXPECTED_COMMIT=$(git rev-parse HEAD) \
  scripts/ref_gh/aurora_feedback_continuation_debug12.pbs
```

The compact logs, JSON, mesh structure, hashes, and status manifest are under
`docs/fo_gh_artifacts/ref_gh_feedback_continuation_20260823/`.  Large CBIN and
restart outputs are intentionally excluded.
