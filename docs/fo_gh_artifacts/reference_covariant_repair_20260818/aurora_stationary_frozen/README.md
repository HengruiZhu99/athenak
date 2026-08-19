# Aurora frozen stationary-trumpet t=1 execution record

## Decision

The current corrected Ref-GH stationary-trumpet **t=1 gate was not assessed**
on Aurora. The sole corrected `dx=1/16` attempt initialized successfully but
ended in a Level Zero GPU write fault at cycle zero, before the first evolved
step. The `dx=1/24` and `dx=1/32` cases were therefore not launched, and no
`t=20` case was submitted. This is an execution/runtime blocker, not evidence
of a Ref-GH formulation instability or of stationary evolution failure.

No source, formulation, parameter, threshold, floor, clipping, reset, or
excision change was made while handling this result. The run used the frozen
source commit `9d55f7b411171aaf2d7e0dc6c3d9be2bfd7ffe0a` and its recorded Kokkos
submodule commit; `source_provenance.txt`, `athena.sha256`, CMake cache, and
build environment provide the complete compact provenance.

## Attempts

| PBS job | Result | Evidence |
| --- | --- | --- |
| 8766889 | Command-line setup rejected before initialization because `ref_gh/source` was not an explicit input parameter. The frozen solver's default is covariant, so the retry omitted this unnecessary override. | `attempt1_command_line_source_rejected.log`, `qstat_8766889.txt` |
| 8766897 | Corrected `dx=1/16`, `tlim=1` attempt exited 134 after a GPU write fault at cycle 0/time 0. | `attempt2_dx16_gpu_write_fault.log`, `qstat_8766897.txt` |

## What did run

The attempt-2 mapping records one MPI rank on `x4302c1s5b0n0`, local rank 0,
with `ZE_AFFINITY_MASK=0.0` and the Intel Data Center GPU Max 1550 Level Zero
backend. Before evolution, the application reported a stationary initial RHS
Linf of `8.55241e-17`, frame reference-Ricci Linf `1.38778e-16`, and completed
its t=0 Ref-GH and common-ADM history rows. The Ref-GH t=0 history has zero
reported GH, reduction, curl, Q, Delta, and bad-state values; its frame-Ricci
and curvature-source Linf values are `2.513374269302154e-16` and
`5.026748538604307e-16`, respectively.

Immediately after the cycle-0 status line, the Intel compute runtime reported
`Segmentation fault from GPU ... type: 0 (NotPresent) ... access: 1 (Write)`
and aborted rank 0. No history row at positive time exists. The compact
attempt-2 Ref-GH and six common-ADM histories are retained here. The roughly
205 MB restart output remains on Aurora and is intentionally excluded from
Git.

## Reproduction boundaries

`scripts/ref_gh/aurora_stationary_frozen_t1_debug.pbs` is the exact corrected
one-node debug launcher. It runs the 64, 96, and 128-cell ladder sequentially
only if the preceding case returns successfully. In this record, the first
case stopped the script, as intended. The campaign must not be advanced to
`t=20` until the first-step GPU fault is reproduced and resolved without
altering the frozen Ref-GH source campaign.
