# Optimization log

## Candidate 1: lean single-tile runtime (`9c98bd14`)

This candidate follows the measured one-PVC profile. All changes are default
off. `z4c/lean_runtime=true` selects the optimized defaults, and each
sub-option remains independently overridable.

| measured overhead | change | evolution arithmetic |
|---|---|---|
| 212 host-staged shared-vertex synchronizations in 21 cycles | precompute canonical group metadata once and perform the same finest-level ordered average and replacement in two device kernels on one rank | same ordered sum and division; no change to synchronization semantics |
| 656 full-state admissibility scans in 21 cycles | retain fail-closed scans immediately before state consumption/projection and at the final accepted-state boundary | observational only |
| vertex-axis record allocation, D2H copy, host audit, and MPI reduction at every enforcement | run the identical axis projection and tolerance rejection on device without materializing the audit table | identical projection; diagnostic localization disabled |
| replay-shadow criterion after the final recorded event | return before criterion evaluation once the replay authority is exhausted | no future topology event exists by authority |
| full-grid source-rate scan for a structurally zero rate | skip when telegraph lapse, shift damping, Z4c damping, and slow-start damping are all absent | exact zero known from configuration |
| timestep-contract and zero-shift CSV diagnostics | disable in the optimized benchmark | observational only |

The optimized benchmark retains O4 finite differences, RK4, CFL `0.15`, KO
`0.50`, P6 native-VC transfer, shock-avoiding lapse, prescribed-zero shift,
zero constraint damping, the replayed frozen hierarchy, and the production
boundary conditions.

### Local qualification before Aurora

- Fresh host build: pass.
- Focused topology, state-admissibility, timestep-contract, AMR-history,
  cadence, and integration tests: 11/11 pass.
- Native-VC record/replay regression with lean mode enabled: pass.
- Lean and exhaustive replay Z4c history files: byte identical, SHA-256
  `b03a17e346ddcef2ade03b79b3ea45f7ab67a3394da3cf50cc96a358d58fcbf5`.
- Legacy input/restart selection through `ATHENA_Z4C_LEAN_RUNTIME=1`: pass;
  the new selector is not materialized into the legacy restart.
- Both 2D Cartoon and 3D Cartesian native-VC AMR-history tests exercise the
  lean path and pass.

A 16-way broad host sweep initially reported six failures. Three were resource
contention/timeouts and passed serially. One candidate fixture omitted the new
default-off declaration and passed after correction. The remaining two output
tests and one exact-base RHS serialization test reproduce with identical hashes
at pre-optimization commit `d35c8248`; their numerical payload/history hashes
remain exact. They are therefore pre-existing stale whole-file metadata goldens,
not candidate numerical regressions.

### Fail-closed Aurora setup attempts

- Job `8790578`: the tiny PVC smoke passed, but the N512 restart rejected a
  command-line parameter absent from the legacy restart header. No N512 PDE
  step executed.
- Job `8790593`: the smoke passed, but the N512 replay rejected the performance
  build's new source ID against the frozen authority. No N512 PDE step
  executed. The next job uses the existing replay-only compatibility assertion
  for the exact authority source ID.

Aurora timing, profile, and numerical-equivalence results are recorded below
after the matched job completes.

### Matched Aurora result

Job `8790595` reached `t=9.85 M` cleanly. Relative to the unmodified job
`8790557`:

| quantity | baseline | candidate 1 | ratio |
|---|---:|---:|---:|
| execution time | 46.29792 s | 19.79745 s | 0.4276 |
| zone-cycles/s | 1,763,040 | 4,123,011 | 2.3386x |
| output wall time | 2.623702 s | 2.778474 s | 1.0590 |

The evolved restart payload and complete Z4c history are bitwise identical.
Candidate 1 therefore delivers a valid `2.34x` speedup, but it does not pass
the required `3x` gate. A second profile is required before another change.

## Subsequent measured candidates

All entries below use the same frozen 212-MeshBlock N512 replay window and one
Aurora PVC.  Candidate 6 is the frozen single-tile implementation used for the
scaling campaign.

| candidate | source | principal change | execution (s) | output (s) | zone-cycles/s | baseline speedup | disposition |
|---|---|---|---:|---:|---:|---:|---|
| 2 | `5c346599` | gather each output variable over all blocks on device | 21.68312 | 4.643783 | 3,764,453 | 2.135x | rejected: slower than candidate 1 |
| 3 | `9af03f4d` | compact P6 and physical-boundary launch domains | 20.25371 | 4.312741 | 4,030,129 | 2.286x | exact but below gate |
| 4 | `4b0e81f0` | fold the 25-variable dimension into each ordered VC neighbor team | 15.90678 | 3.702279 | 5,131,466 | 2.911x | exact; within 3% of gate |
| 5 | `abba38ea` | fail closed directly on device after lean P6 | 24.97885 | 13.20828 | 3,267,768 | 1.853x | exact; filesystem/output outlier |
| 6 | `02a9b465` | one full-source mirror per homogeneous lean output block | 13.91421 | 2.164351 | 5,866,310 | **3.327x** | **hard gate pass** |

Candidate 2 was reverted before further work.  It removed small-copy count but
introduced 48 large range kernels with expensive index decomposition; both its
output time and end-to-end time worsened.  Candidate 6 instead mirrors each
homogeneous source view (`u0`, `u_con`, or the derived diagnostic view) once
and performs the exact active-range extraction on host.  The local production
output fixture was byte-identical between lean and historical paths for state,
ADM, constraints, and diagnostics.

The candidate-3 profile quantified the compact-P6 effect: 21-cycle P6 device
time fell from `0.879892 s` to `0.654166 s`.  The four VC pack/unpack kernels
remained essentially unchanged at about `0.956 s` combined, which motivated
candidate 4.  Candidate 5's non-output time was `11.77057 s`, versus
`12.20450 s` for candidate 4, so the direct device failure gate saved about
`0.434 s`; its poor end-to-end result was entirely dominated by a
`13.20828 s` output outlier.

Candidate 6's non-output time is `11.74986 s`, a `3.72x` reduction from the
baseline's `43.67422 s`.  Its complete end-to-end throughput is also 3.5%
above the earlier Perlmutter A100 observation of approximately
`5.67e6 zone-cycles/s`.  This is a performance comparison only, not a hardware
generality claim.

## Multi-rank candidates and final-source rerun

Source `4260e5ba` first removed an observational inconsistency: lean MPI runs
still executed the exact shared-node postcondition and its device/host/MPI
reduction after every reconciliation. Keeping the default exhaustive path
unchanged, honoring the lean opt-out improved the measured non-output portion
by 4.6% at 2 tiles and 10.1% at 24 tiles. It did not restore scaling because
field values still followed the global host-staged `MPI_Allgatherv` path.

Source `62993e7b` replaced only that lean MPI hot path with precomputed sparse
owner/participant communication through persistent device buffers and
GPU-aware point-to-point MPI. The owner retains the exact canonical
finest-level summation order. Aurora job `8790725` measured `1.503x` higher
throughput at 2 tiles and `1.588x` at 24 tiles relative to the matched
postcondition-only cases. Both histories and final restart payloads were
bitwise exact.

The required post-change one-PVC rerun, job `8790731`, reached
`5.958188e6 zone-cycles/s` in `13.69965 s`, including `1.962092 s` of output.
This is a `3.37950x` end-to-end speedup over the unmodified `1.763040e6`
baseline. Its complete history and final restart numerical payload are
bitwise identical to the baseline. The hard `3x` gate therefore passes on the
final source; the `5x` stretch gate does not.
