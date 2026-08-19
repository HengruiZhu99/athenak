# Aurora frozen stationary-trumpet t=1 execution record

## Decision

The current corrected Ref-GH stationary-trumpet **t=1 gate was not assessed**
on Aurora. Two corrected `dx=1/16` attempts on different Aurora nodes both
initialized successfully and ended in a Level Zero GPU write fault at cycle
zero, before the first evolved step. The `dx=1/24` and `dx=1/32` cases were
therefore not launched, and no `t=20` case was submitted. This is a
reproducible execution/runtime blocker, not evidence of a Ref-GH formulation
instability or of stationary evolution failure.

No source, formulation, parameter, threshold, floor, clipping, reset, or
excision change was made while handling this result. The run used the frozen
source commit `9d55f7b411171aaf2d7e0dc6c3d9be2bfd7ffe0a` and its recorded Kokkos
submodule commit; `source_provenance.txt`, `athena.sha256`, CMake cache, and
build environment provide the complete compact provenance.

## Attempts

| PBS job | Result | Evidence |
| --- | --- | --- |
| 8766889 | Command-line setup rejected before initialization because `ref_gh/source` was not an explicit input parameter. The frozen solver's default is covariant, so the retry omitted this unnecessary override. | `attempt1_command_line_source_rejected.log`, `qstat_8766889.txt` |
| 8766897 | Corrected `dx=1/16`, `tlim=1` attempt on `x4302c1s5b0n0` exited 134 after a GPU write fault at cycle 0/time 0. | `attempt2_dx16_gpu_write_fault.log`, `qstat_8766897.txt` |
| 8766926 | Identical corrected retry on `x4303c1s3b0n0` exited 134 after the same cycle-0 GPU write fault. | `attempt3_dx16_gpu_write_fault.log`, `qstat_8766926.txt` |

## What did run

Both mappings record one MPI rank, local rank 0, with `ZE_AFFINITY_MASK=0.0`
and the Intel Data Center GPU Max 1550 Level Zero backend. Before evolution,
each application instance reported stationary initial RHS Linf
`8.55241e-17`, frame reference-Ricci Linf `1.38778e-16`, and completed its
t=0 Ref-GH and common-ADM history rows. The two Ref-GH histories are
byte-identical (SHA-256 `45b5c65aee52af534497073fd076a79d89a08e7a0dd7c27993070cae7ff0d4d3`);
the common-ADM global histories are also byte-identical (SHA-256
`427d2fac28f0084619193061923d7f5b19fdc4facdc04a9b53c7150edda8aab2`).
The Ref-GH t=0 history has zero reported GH, reduction, curl, Q, Delta, and
bad-state values; its frame-Ricci and curvature-source Linf values are
`2.513374269302154e-16` and `5.026748538604307e-16`, respectively. The
machine-readable matched t=0 comparison is `t0_matched_history_comparison.tsv`.

Immediately after the cycle-0 status line, each Intel compute-runtime log
reported `Segmentation fault from GPU ... type: 0 (NotPresent) ... access: 1
(Write)` and aborted rank 0. No history row at positive time exists. The
compact Ref-GH and six common-ADM histories for both attempts are retained
here. The roughly 205 MB restart output from each attempt remains on Aurora
and is intentionally excluded from Git.

## Reproduction boundaries

`scripts/ref_gh/aurora_stationary_frozen_t1_debug.pbs` is the exact corrected
one-node debug launcher; `aurora_stationary_frozen_t1_retry.pbs` is its
separately preserved identical retry with a new output path. Each runs the
64, 96, and 128-cell ladder sequentially only if the preceding case returns
successfully. In both records, the first case stopped the script, as intended.
The campaign must not be advanced to `t=20` until the reproducible first-step
GPU fault is resolved without altering the frozen Ref-GH source campaign.
