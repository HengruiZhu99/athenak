# Original-driver n32 control: checkpointed, scientific checks failed

Only **n32** ran in this allocation. The native executable stopped at a
bounded checkpoint after **929 accepted steps**, at coordinate
**t = 0.23225**, before the requested t = 0.6 endpoint. It returned code 0
and `status="checkpointed"`; its saved V1 checkpoint records `admitted=1`.
The worker nevertheless classified the case and campaign as **failed**
because the state, Hamiltonian and momentum diagnostics exceeded the
reviewed scientific thresholds. Native checkpoint admission does not
override these failed scientific checks. **This package admits no restart,
stability result, or convergence result.**

| Current diagnostics at t = 0.23225 | Observed | Reviewed limit |
| --- | ---: | ---: |
| Maximum stored-field deviation | 10.1816494972 | 1e-8 |
| Reported Hamiltonian L2 | 1.0441964403e-3 | 1e-7 |
| Momentum L2, Penrose norm | 1.1405263649e-1 | 1e-7 |
| Surface null running maximum | 1.5543122299e-15 | 1e-12 |
| Interpolated-ghost null running maximum | 1.5543122345e-15 | 1e-12 |

The final diagnostics are current, unlike the earlier n16 numerical
rejection. Here the native report contains no geometry-rejection error;
the reported minimum chi and lapse remained positive, at 0.9379797661
and 0.5000228882. Precise null closure coexists
with large equilibrium and constraint errors. The requested endpoint was
not completed, and a bounded native checkpoint must not be described as a
successful stability run.

This is the unchanged original driver: n = 32 per block, **622,592 active
volume cells**, dt = 0.00025, 12 SYCL MPI ranks on 12 distinct GPU tiles,
and the fixed 19-block Cartesian-core/spherical-shell mesh. The binary
SHA-256 is
`6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef`;
the numerical source-manifest SHA-256 is
`3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f`.
No algebraic-projection or cross-face angular candidate is used.

The n24 and n16 half-timestep cases in the [plan](plan.json) were **not
started** after the worker stopped. They were submitted separately in
`resolution-control-v2`, follow-up job **8982461**; their outcomes and
records are not part of this package. The combined growth-rate, timestep
and throughput comparison remains separate. This n32 result alone does
not establish a convergence order or uniquely identify the unstable
operator. The [protocol](protocol.md) and [analysis tool](analyze.py) are
maintained separately and were not changed or run while packaging this
allocation.

Job **8982298** was collected with PBS state **F**, scheduler exit status
**1**, and its allocation lock was released. These scheduler/worker
outcomes are distinct from the native executable's checkpointed status
and return code 0. [terminal-status.json](terminal-status.json) retains
only selected terminal/release fields and the hashes of their original
records, omitting raw scheduler account and mail metadata.

The retained [verification](verification.json) is the immutable full-payload
verification made during collection: all three worker-recorded hashes and
byte counts match the report, history and **311,853,128-byte** checkpoint.
The checkpoint payload is omitted. Its verified header identifies
`HYP3D_MPI_V1`, n32, 929 steps, t = 0.23225, dt = 0.00025 and `admitted=1`.
[package-verification.json](package-verification.json) separately rechecks
the retained small artifacts and plan/PBS/source/binary bindings; it does
not claim a second full-payload hash pass.

The package retains the original worker status, report, history, plan,
binding, PBS launcher, affinity/execution logs, and verification, plus a
concise [assessment](assessment.json). It contains no executable, full
checkpoint payload or raw scheduler/account records. `package-files.json`
inventories these package additions, excluding itself and the separately
maintained `analyze.py` and `protocol.md`.
