# Aurora Ref-GH performance-portability evidence, 2026-08-20

## Decision

The Aurora PVC full-output evolved-cycle gate is **not passed**.  The current
equation-preserving portability refactor builds and its CPU history output is
byte-identical to the pre-refactor output, but PBS job `8769672` exits 134 on
the first evolved step.  All Ref-GH history kernels and explicit Kokkos fences
complete before the next `CalcRHS zero` write reports a Level Zero
`NotPresent` GPU page fault.  No three-resolution convergence run was launched,
and this evidence establishes neither convergence nor long-time stability.

The remote campaign root is
`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_aurora_stationary_frozen_20260819`.
Runtime logs and provenance below are copied verbatim; restart files are
excluded and trailing blanks in generated history headers are normalized.
`qstat.txt` was captured after completion.  No Aurora Ref-GH jobs were left
queued or running.

## Decisive evidence

| PBS job | Result | Interpretation boundary |
| --- | --- | --- |
| `8769259` | Both intermediate metric-condition substages exit 134. | Historical localization evidence only; its stage numbering belongs to an intermediate source state. |
| `8769485` | Read-only condition-number reduction fences, then the next evolved RHS write faults. | Removing history-side staging writes was insufficient. |
| `8769542` | Condition number is cached during diagnostics; every history reduction fences, then the next evolved RHS write faults. | Moving the eigensolve out of history was insufficient. |
| `8769596` | Stage-4 and stage-5 bounded one-cycle cases both exit 0. | The cached-condition kernel and its then-trivial reduction can each complete in a truncated history path. This does not qualify the full path. |
| `8769628` | One custom batched maxima reduction fences, then the next evolved RHS write faults; exit 134. | Reducing history kernel count with a custom array reducer was insufficient. |
| `8769672` | Three built-in combined `Kokkos::Max<Real>` reductions fence, then the next evolved RHS write faults; exit 134. | This is the current-branch decisive result and direct mature-AthenaK reduction-pattern comparison. |

The final log is
`job_8769672/one_cycle/ref_gh_stationary_portability_dx16.log`, SHA-256
`e627b0ce28f8c2d10e87f9911ffb1fab785c18e5bb2ab0aabdb3a94d4cc1f621`.
Its build, executable/source provenance, GPU mapping, histories, and complete
PBS record are in the same job directory.

## Local equivalence check

The current built-in combined-reducer source was compiled in the isolated
worktree `/tmp/athenak-refgh-portability` and completed a one-cycle `32^3`
Kokkos-Serial stationary-trumpet run.  Its native history was byte-identical to
the scalar-reduction reference:

```text
reference: /tmp/refgh-cache-condition-cpu-wWgLQv/cache_condition_cpu.ref_gh.hst
candidate: /tmp/refgh-combined-max-cpu-k1mZL0/cache_condition_cpu.ref_gh.hst
both SHA-256: 30e568589b325c6857d5ace9040ad2da81ec5069598e42da0da5771ee78d81d6
ADM history set comparison: byte-identical
```

The local run ended at cycle 1 with field Linf `1.665335e-15`, constraint Linf
`2.042604e-14`, and no nonfinite state.  This verifies output equivalence on
CPU only; it does not weaken or supersede the failed PVC gate.

## Current source state

The committed source retains only equation-preserving changes: caller-owned
reference-geometry outputs, smaller RHS kernels, compact Psi kinematics,
diagnostic condition caching, built-in combined history maxima reductions,
batched MPI sum/max history reductions, and default-off fences/repeat-RHS
instrumentation.  The mathematical formulation, finite-difference stencils,
RK algorithm, dissipation, gauge/source equations, and acceptance thresholds
are unchanged.

The ready-to-send review/debugging prompt is
`docs/ref_gh_aurora_portability_handoff_prompt.md`.
