# Original-driver n24 and n16 half-timestep controls: both failed

Both Minkowski cases rejected `nonpositive interpolated scri geometry`
before the requested coordinate t = 0.6 endpoint. The n24 case reached
**t = 0.3136666667 after 941 accepted steps**; n16 with half the original
timestep reached **t = 0.486 after 1,944 accepted steps**. Both saved V1
checkpoints have **`admitted=0`**. This package admits no restart, stable
evolution, or convergence result.

| Case | dt | Last accepted t | Last complete diagnostic t | Launcher return code |
| --- | ---: | ---: | ---: | ---: |
| n24 | 1/3000 | 0.3136666667 | 0.300 | 2 |
| n16, half dt | 0.00025 | 0.486 | 0.475 | 143 |

The report diagnostics are stale at the final accepted times:
`diagnostics_current=false` in both cases. The following values belong to
the last complete diagnostic times above, not the rejection endpoints.
The histories retain those times without extrapolation.

| Last complete diagnostics | n24, t = 0.300 | n16 half dt, t = 0.475 |
| --- | ---: | ---: |
| Maximum stored-field deviation | 4.8742126655 | 12.7179264245 |
| Reported Hamiltonian L2 | 4.5451802157e-4 | 8.2007415556e-3 |
| Momentum L2, Penrose norm | 6.4531601463e-2 | 4.0151991143e-1 |
| Minimum chi | 0.9757627746 | 0.8633245610 |
| Minimum lapse | 0.5000406901 | 0.5000915527 |

Constraint norms use normalized compactified Cartesian coordinate volume.
The reported Hamiltonian is the evolved-variable formulation; it has not
been replaced by a reconstructed off-constraint ADM Hamiltonian. The
report's running surface/ghost null maxima remain at about 1e-15 in both
cases, while state and constraint errors grow. Precise null closure does
not establish stable evolution or identify the growing operator.

The launcher outcomes require separate interpretation. For n24, the
worker records return code 2 and explicitly approved continuation after
the recognized numerical failure, with its unadmitted checkpoint as
evidence. This allowed the n16 half-timestep case to run. For n16, the
[log](output/minkowski-n16-half-dt.log) ends with native
`status=failed t=0.486 error=nonpositive interpolated scri geometry`, then
reports **rank 11 exited with code 2** and **rank 4 died from signal 15**.
The launcher returned **143**; the worker records `timed_out=false` and
`storage_error=null`. These facts establish numerical rejection and a
mixed-rank shutdown. They do not establish a uniform native exit code or
the cause of signal 15. **Only n24** is recorded as continued after failure;
no guarded-continuation claim is made for n16.

This allocation uses the unchanged original driver, the fixed 19-block
Cartesian-core/spherical-shell mesh, 12 SYCL MPI ranks and 12 distinct
GPU tiles on one Aurora node. The n24 grid has 262,656 active volume cells;
n16 has 77,824. The numerical binary SHA-256 is
`6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef`;
the source-manifest SHA-256 is
`3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f`.
The [preflight source-equivalence record](../preflight-v2/source-commit-equivalence.json)
binds this to numerical commit `113aaff783008204e1ff8cbf2fb8bb5f2af2204c`.
Neither algebraic projection nor cross-face angular exchange is used.
The original n16 baseline and n32 control belong to
[stability-v1](../stability-v1/README.md) and
[resolution-control-v1](../resolution-control-v1/README.md), respectively.
The combined growth-rate and timestep comparison is maintained separately.

Job **8982461** was collected with PBS state **F**, scheduler exit status
**1**, and worker status **failed**. Its allocation was released at
2026-10-29 05:33:32.739951 UTC. The sanitized
[terminal record](terminal-status.json) preserves selected terminal/release
fields and source hashes; raw scheduler account and mail metadata are
omitted.

The immutable [verification](verification.json) records matching hashes and
byte counts for all **eight** worker artifacts, including both full
checkpoints and both failed-stage captures. Both checkpoint headers
confirm `admitted=0`. [Package verification](package-verification.json)
rechecks the four retained report/history files, all eight artifact
records, the source/binary/plan/PBS bindings and all 36 worker report
checks. It does not claim a second full-payload hash pass.

After both control allocations were released, the
[purge manifest](purge-resolution-controls-manifest.json) and
[receipt](purge-resolution-controls-receipt.json) record removal of exactly
five verified remote payloads: the earlier n32 checkpoint and this
allocation's two checkpoints plus two failed-stage captures. Full local
copies were preserved and verified; remote hashes were rechecked before
removal. The removed logical payload size was **568,884,440 bytes**;
charged campaign footprint fell from 1,523,638,272 to 954,722,012 bytes.
No other remote files were deleted. Original worker artifact records are
preserved as records of the archived payloads.

This small package retains original reports, histories, logs, affinity,
plan, binding, worker status, PBS launcher and verification, with an
[assessment](assessment.json). It excludes executables, full numerical
payloads and raw scheduler records. `package-files.json` inventories the
retained files, excluding itself.
