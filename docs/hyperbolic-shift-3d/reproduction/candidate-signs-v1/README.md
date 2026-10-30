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

Job8990333 continues with its original frozen executable. It cannot gain this
instrumentation while running. Use the instrumented binary for the next healthy
checkpoint continuation, or an explicitly identified diagnostic replay from the
verified3M input if that trial fails; never restart its unadmitted failure state.
Do not label uninstrumented portions as having zero negative candidates.

Validation: unit cases cover negative/zero/positive/nonfinite candidates,
separate stage/final counts, extrema and first times, and exposed chi underflow
without changing clipping. A two-rank n16 puncture step from3M produced a
checkpoint bitwise identical to the uninstrumented AL1 run, including RK
compensation. The new report correctly contained an empty event list there.
