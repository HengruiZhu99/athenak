# Puncture chi-floor retry toward 20M

The previous n16 run failed at 9.141M when chi became zero in the eight cells
nearest the puncture. The user requested a floor and a new retry toward 20M.

This separate target adds chi_min=1e-8 only for coordinate radius r<.03, inside
the initial coordinate horizon (~.063). It clamps finite log-chi candidates
at every volume RK stage and the final compensated update, and clears chi's
compensation when the final update is clipped. It does not repair NaN or infinite
logarithms, change the lapse floor, apply a floor at scri, or change RHS equations.
A new check rejects nonfinite logarithmic rates after division by chi/lapse.

The value is an initial numerical regularization choice, not a demonstrated
converged prescription. The retry must establish whether it prevents failure;
a completed run would still need exterior accuracy and floor-sensitivity checks.
No claim of long-time stability is made at submission.

Counters distinguish intermediate-stage and final-update interventions and
record each affected global cell, its radius and total hits. They include
attempted stages of a rejected step and reset at each process/restart segment.

Checkpoint schema HYP3D_MPI_CF1 appends the exact floor and support radius to
its header (168 bytes). The importer accepts only the previously verified 3M
checkpoint hash, changes its format/configuration explicitly, and preserves
all evolved fields and RK compensation bytes. No failed 9.141M checkpoint is
admitted or used as a restart. New checkpoints require the exact floor policy.

Run n16 from 3M to 20M at dt=.0005, dt/dx=.0444444. Keep the original inactive
scri/angular caps, horizon lmax=12/ntheta=24, diagnostic interval .05 coordinate
time, 12 ranks/GPU tiles on one CompactBinaryMerger debug node, one-hour walltime,
SYCL+MPI and make -j64. Use fresh source/build/plan/output directories.
Re-enable the existing 30-minute monitor after successful submission. Resume
only healthy admitted walltime checkpoints with unchanged CF1 policy and source.
Stop and report numerical failure. Keep remote home footprint below 5 GB and
hash-verify local evidence before purging large remote outputs.
