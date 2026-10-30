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

## Validation and restart protocol

The unit tests passed. A two-rank local one-step comparison from 3M to 3.0015M
has bitwise-identical evolved fields and RK compensation to the original DT1
executable, with zero floor interventions. A subsequent CF1 restart passed;
wrong-floor and unmigrated DT1 checkpoints were rejected. See local-parity.json
and restart-tests.json. These short local checks do not test the late puncture
failure or establish GPU long-time stability.

Frozen scientific source: 95e1d54029517a1532d7aecc0fb5576cee1a6ee2.
Remote root: /home/hzhu/hyperbolic-core-shell-20261028.
Source: source-chi-floor-v1; manifest: source-chi-floor-v1-manifest.json.
Controller: source-chi-floor-v1/tst/hyperbolic_shift_3d/aurora/control.py.
Binary: build-sycl-mpi-chi-floor-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_puncture_chi_floor.
First plan: campaigns/n16-chi-floor-to20m-v1-segment01/plan.json.
The imported checkpoint SHA256 is
 a4685b88a6b7c89baa24117617981b5114c84934d322aee31824d2aab703df7b.

For each automatic continuation, require a terminal healthy walltime stop and
an admitted CF1 checkpoint: n=16, nf=31, kind=2, admitted=1, dt=.0005,
finite time in [1,20/3), length 168+16*(31*cells+surface). The last two doubles
of the 168-byte header must be chi_floor=1e-8 and radius=.03. Match checkpoint
header time/steps/admission to its report and verify artifact SHA256. Retain
source and executable hashes, all coefficients, horizon lmax=12/ntheta=24,
and diagnostic interval .05. Carry a horizon-seed sidecar only when bound to
the matching checkpoint. Floor counts are per segment and must be summed
across segments for campaign totals. Preserve horizon quality flags.

Collect the terminal plan and release its matching root lock with
collect --release-terminal-lock. Check all user debug jobs, prepare and bind
a fresh sequential segment plan, and submit only when no allocation is active.
Do not resume a numerical, storage or ambiguous scheduler failure. At verified
20M completion collect the scientific evidence and pause the monitor. Copy and
hash-verify large artifacts locally before purging remote outputs; retain any
inputs still needed by queued/running successors. Keep the remote footprint
below 5,000,000,000 bytes and preserve unrelated jobs and files.

## Submission

Aurora build passed with SYCL/MPI and make -j64 (172.34 seconds). Job **8989676**
was submitted on one CompactBinaryMerger debug node with 12 MPI ranks/GPU tiles.
The existing 30-minute monitor was activated at submission. It was removed
after the numerical failure reported below; no further retry is scheduled.
The campaign footprint at binding was 3,968,929,760 bytes, below the 5 GB limit.
See build-status.json, binding.json, submission.json and monitor.json for the
exact hashes, resources and monitoring policy. No long-time result is yet claimed.

## Result: floor did not establish stability

Job 8989676 failed at coordinate t=3.052 = **9.156M**, after 4104 new accepted
steps (6105 total), with `invalid accepted volume geometry`. Native elapsed time
was 906.79 seconds. This is ten steps / .015M beyond the original no-chi-floor
failure at 9.141M. There was no walltime or storage failure, and no resubmission.
The checkpoint is explicitly unadmitted and must not be used for restart.

SHA256-verified failed-state inspection found positive chi everywhere, no
nonpositive conformal-metric eigenvalues, and **positive-infinite lapse in the
eight nearest puncture cells**, at all combinations of coordinates ±.005625.
These are the same cells where the original run lost chi positivity. Chi in
the last accepted state equals 1e-8 there; the rejected candidate has chi about
.04349. The only nonfinite volume field is lapse (field18). Thus the chi floor
prevented the immediate zero-chi failure but the coupled update still diverged;
a lower lapse floor cannot prevent this positive overflow. The recorded last
RHS maximum is 1.83359e21. These observations do not yet identify the upstream
cause of that growth.

There were 216 intermediate-stage chi clips and 104 final-update clips across
32 cells, all at r<=.0186561; counts include the rejected attempted step. There
were no lapse-floor interventions. Maximum scri null residuals remained
1.22126e-15 (surface) and 1.55431e-15 (ghost). The last complete ordinary
diagnostic was at9.15M, not9.156M: H_L2=2.73304, minimum lapse=.0209898.
The horizon candidate radius=.1066379 has expansion RMS=.00287336, above the
.001 acceptance threshold, so accepted exterior constraint norms remain absent.

The terminal root allocation lock was released. Large remote failed-state and
checkpoint files were deleted only after verifying local copies against the
worker SHA256 records; small evidence is retained here and raw data locally in
output/hyperbolic-core-shell/chi-floor-v1/segment01. See verification.json and
verified-cleanup.json. The automatic retry monitor was deleted after this
numerical failure. No further parameter changes or evolutions were launched.
