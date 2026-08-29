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
