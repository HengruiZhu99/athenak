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

Aurora timing, profile, and numerical-equivalence results are recorded below
after the matched job completes.
