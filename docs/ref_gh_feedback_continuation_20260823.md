# Ref-GH feedback continuation, 2026-08-23

## Status

This branch implements the equation-preserving feedback clock requested in the
controlling goal.  Local T0--T2 gates pass, including exact prescribed-law
equivalence and restart continuity.  The required Aurora tau-8 replay has not
run, so the risk thresholds below remain **candidates**, T3 has not started,
and no closed-loop continuation or stability claim is made.

The work is based exactly on
`9c438dc619aa742404530c953243d71b2a01d8e6` from
`codex/ref-gh-transition-path-ordering-diagnosis-20260822`.  The implementation
branch is `codex/ref-gh-feedback-continuation-20260823`; the controller code and
inputs are commit `9b86a5d97af9abb46b18c82c4895feb9887f0d5d`, with the
stronger manufactured-history audit in
`55eed5d4267d2eea833c4fe3ec6e8ff3af3bc732`.

Aurora authentication was restored and debug job `8777229` ran on node
`x4311c6s7b0n0`.  It proved twelve distinct PVC tile mappings (`0.0` through
`5.1`), built the intended Kokkos `SERIAL;SYCL`/PVC/MPI executable, and wrote
the exact 272-block mesh tree.  The direct mesh-only process then returned 139
during SYCL teardown before any controller test ran.  This is a launcher-gate
failure, not solver evidence.  Following the mature Aurora pattern, one
equation-preserving script correction maps mesh-only mode through one MPI tile
and accepts 139 only after the complete expected tree is verified.  A single
focused rerun is pending; no numerical or controller source was changed.  The
launcher correction is commit `0944be6b18dcc89f04a1e98fb34dfaef2a18622d`.

The corrected job `8777250` then passed the mapped mesh audit, the strengthened
controller unit test on all twelve ranks, and both one-cycle PVC evolutions.
An off-node comparison proves their CBIN payloads are bitwise equal
(`Linf=0`).  The job stopped before restart testing because Aurora's default
`python3` is 3.6.15 and rejected `from __future__ import annotations` in the
analysis script.  Aurora also provides `/usr/bin/python3.10` with NumPy 2.2.6;
the harnesses now pin and record that interpreter in
`7a69477bec06f426c6f105300b724bd25977ce45`.  This is a harness portability
fix only.  Job `8777250` is therefore partial evidence, not a passed gate.

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

## Safety policy and candidate thresholds

The hard absolute caps are enforced in the parser:

| Quantity | candidate stop | immutable cap |
|---|---:|---:|
| `kappa_G` | 8 | <= 8 |
| `a_min` | 0.5 | >= 0.5 |
| `a_max` | 3 | <= 3 |
| `v2_max` | 0.20 | <= 0.20 |

The risk ramp currently slows at `R=0.70` and stops at `R=1`.  These values are
not frozen scientifically until the mandated old-medium tau-8 replay records
its envelope.  They must not be retuned after T1 replay or after observing a
closed-loop outcome.

The native-constraint warning levels are GH L2 `2e-2`, reduction L2 `5e-3`,
and curl L2 `8e-2`; smaller values are rejected.  A warning commands
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

## Remaining ordered gates

1. Complete one twelve-tile debug job with
   `scripts/ref_gh/aurora_feedback_continuation_debug12.pbs`; do not leave a
   competing request.  Jobs `8777229` and `8777250` preserve the mesh-launcher
   and Python-harness failures; the latter proves controller/PVC execution and
   exact prescribed equivalence but did not reach restart testing.
2. Pass the PVC T0--T2/restart gate, then replay the existing medium fixed-core
   tau-8 run to t=4M with
   `scripts/ref_gh/aurora_feedback_tau8_replay12.pbs` and freeze thresholds
   from its new diagnostics.  The replay script launches no feedback case and
   emits a fail-closed threshold-freeze decision for review.
3. Run T3 cheap medium only to `xi>=0.5` or t=5M.
4. Run T4 on the enlarged medium domain through xi=1 plus a 2M hold, t=20M, or
   fail closed; then T5 aggressive prescribed four-M discriminator.
5. Run the three-resolution T6 gate only if T4 passes.

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
