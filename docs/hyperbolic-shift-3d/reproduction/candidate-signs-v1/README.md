# Pre-floor candidate sign logging

The user explicitly selected keeping the existing floors and logging negative
candidates. The AL1 numerical policy is unchanged: direct lapse RK/KO, log-chi
RK/KO, physical lapse floor1e-10, chi floor1e-8 within r<.03. Finite undershoots
are clipped and evolution continues. Nonfinite states remain numerical failures.

The existing bulk RK stage/final kernels record finite nonpositive volume
candidates before clipping. No extra per-cell launches or communication occur
inside RK. Per-cell/field records distinguish negative and zero candidates,
stage and final counts, minimum candidate, coordinates, and first occurrence
times. Lapse values are the conformal lapse (not alpha-A); times are coordinate
times. Log-chi exponentiation cannot yield a finite negative value, but zero
from underflow is recorded separately. A negative logarithm is not a negative
physical field and is not counted.

Counters cover attempted stages of this process/segment, including a rejected
step, and reset on restart. They are diagnostics only, absent from AL1 checkpoint
payloads. Summaries are emitted in candidate_sign_events at segment completion
or failure; no-event rows are omitted and absent first-negative/first-zero times
are null. Only event rows are gathered across ranks. Existing total floor hits
still include positive candidates below the floor; those are not negative hits.

Job8990333 used its original frozen executable and failed at9.0975M. The new
instrumented diagnostic replay starts from the verified3M input, never from its
unadmitted failure state.
Do not label uninstrumented portions as having zero negative candidates.

Validation: unit cases cover negative/zero/positive/nonfinite candidates,
separate stage/final counts, extrema and first times, and exposed chi underflow
without changing clipping. A two-rank n16 puncture step from3M produced a
checkpoint bitwise identical to the uninstrumented AL1 run, including RK
compensation. The new report correctly contained an empty event list there.

A synthetic two-rank final-update test deliberately changed the compensation
at cell4095 to create zero chi and negative conformal lapse before flooring.
The native run recorded one zero-chi final event and one negative-lapse final
event (minimum-.9906041013 at coordinate time1.0005), clipped both, and completed
the step. Empty ranks were handled by the sparse event gather. This is a
reporting/floor-path test, not evidence of physical evolution stability.

Frozen instrumented source:1a2cd441ca0cdf04d58116fcdf891b3781f3a35a.
Remote root:/home/hzhu/hyperbolic-core-shell-20261028.
Source:source-candidate-signs-v1; manifest:source-candidate-signs-v1-manifest.json.
Binary:build-sycl-mpi-candidate-signs-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_direct_lapse.
Controller:source-candidate-signs-v1/tst/hyperbolic_shift_3d/aurora/control.py.
The input/checkpoint format and experimental policy remain AL1 and
n16_direct_lapse_retry_to_20M. A next segment must bind this source/binary
explicitly; never mutate the old frozen source or an already-bound plan.

Two obsolete DT1/CF1 imported restart copies were removed remotely after
rechecking local and remote SHA256/size and confirming the active allocation
uses neither. The active AL1 input was retained. See obsolete-inputs-cleanup.json.


## Instrumented replay submission

The SYCL/MPI build passed with make -j64 in194.18 seconds. Job8990381 was
submitted from campaigns/n16-candidate-signs-to20m-v1-segment01/plan.json on
one CompactBinaryMerger debug node with12 MPI ranks/GPU tiles. It uses the
same3M AL1 input, unchanged floors/RK/equations/timestep, and target20M. The
purpose is to expose candidates before clipping; it is not a stability repair.
The existing30-minute monitor now follows this job, continues only healthy
walltime checkpoints, and stops after numerical failure or verified20M completion.
No further automatic fresh numerical-failure replay is scheduled. Candidate
counts from earlier uninstrumented runs remain unavailable. Exact hashes and
policy are in build-status.json, binding.json, submission.json and monitor.json.


## Instrumented result and newly authorized fallback

Job8990381 failed at9.0975M with positive-infinite chi in the same three cells
as job8990333. Both checkpoint and failed-stage payloads are bitwise identical
to that uninstrumented run. The checkpoint is unadmitted and is not a restart.
The sign log records88 stage and69 final zero-chi candidates, first at9.09M;
there are no finite negative chi candidates in the logarithmic chart. Eight
finite negative lapse candidates occurred in the rejected final update at9.099M,
with minimum about-3.745e53. They were floored; they were not the termination
criterion. The last accepted time9.0975M differs from the rejected candidate
update time9.099M. These enormous candidates show growth beyond a small sign
undershoot. The original chi floor counts also include positive below-floor
values. See segment01/candidate-totals.json and verification.json.

After this failure the user authorized one fresh direct-chi/direct-lapse trial,
retaining floors and counting negative candidates without terminating on sign.
See ../direct-chi-lapse-v1. This supersedes the earlier no-further-replay monitor
instruction. The scientific source and results of this replay remain frozen.
