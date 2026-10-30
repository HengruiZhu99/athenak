# n16 continuation from 3M toward 20M

Continue the exactly reproduced 3M checkpoint at dt=.0005 (dt/dx=.0444444),
with unchanged frozen SYCL/MPI executable and numerical coefficients.
Target coordinate time is 20/3 for M=1/3. Checkpoint SHA256:
628c2109bdf7ca6624afff58b65e8a28baca724e9e73de17b517454b26fe6a31.

Use one CompactBinaryMerger debug node, 12 MPI ranks/GPU tiles, one-hour walltime.
Native worker deadline leaves time for an admitted checkpoint. The user requests
30-minute monitoring and automatic resubmission after a healthy walltime stop.
Do not restart a numerical failure or an unadmitted checkpoint. All segments use
fresh output/plan directories and the same source/executable hashes. Keep home
campaign footprint below 5 GB; copy and hash-verify large evidence before purge.

Remote root: /home/hzhu/hyperbolic-core-shell-20261028.
First plan: campaigns/n16-to20m-v1-segment01/plan.json.
Controller: launcher-timestep-v3/control.py; source manifest:
source-timestep-v1-manifest.json. No evolution rebuild is required.

Each monitor run should inspect saved submission/worker records and qstat -fx.
Do not duplicate an existing queued/running job. A terminal checkpointed run can
resume only after validating header HYP3D_MPI_DT1, n=16, nf=31, kind=2, admitted=1,
dt=.0005, finite time in [1,20/3), intact length 152+16*(31*cells+surface), and
artifact SHA256. Match native report/header time, steps, and admission. Keep
source, binary, dt, horizon settings and diagnostic interval unchanged. Carry
an existing bound horizon-seed sidecar only if it matches the checkpoint.
Release the terminal plan's root lock using collect --release-terminal-lock;
prepare a new sequential segment plan, bind, and submit. Numerical, storage, or
ambiguous scheduler failures require inspection; never treat them as walltime.
Check all user debug jobs before submission and avoid concurrent allocations.
On 20M completion collect and verify evidence, report results, pause the monitor.
If SSH expires, request renewal without repeated submissions.

## Result: numerical failure at 9.141M

Job 8989499 terminated with `invalid accepted volume geometry`, after 4094
additional accepted steps from 3M (6095 total), at coordinate t=3.047. Native
runtime was 896.906 seconds. It did not reach 20M or the walltime limit. The
failed run's checkpoint is explicitly unadmitted, and was not resubmitted.
The 30-minute monitor was paused on inspection.

All recorded artifacts were SHA256-verified after local collection. In the
rejected volume candidate, chi is exactly zero in the eight closest cells at
coordinates (±.005625, ±.005625, ±.005625), radius .00974278579. Their chi values
in the last accepted checkpoint were about 3.93884e-7. No other nonpositive chi,
nonpositive lapse, nonfinite field, or nonpositive conformal-metric eigenvalue
was found. Chi is materialized by exponentiating its evolved logarithm, so the
zeros arise in that conversion. The saved physical fields alone cannot
distinguish exponential underflow from a nonfinite logarithmic increment;
the update still needs diagnosis. Do not infer that a chi floor would repair the instability.

Original numerical penalty coefficients remained unchanged (scale=1). Maximum
scri null residual was 1.22126e-15; ghost residual 1.55431e-15. There were zero
lapse-floor interventions. The last complete ordinary diagnostic is at 9M,
not the failure time: min lapse .0196473, full-volume Hamiltonian RMS .113289.
The last complete horizon candidate has radius .1065993 and expansion RMS
.00282462, exceeding the .001 acceptance threshold; accepted exterior norms
are therefore absent. This failure does not establish the horizon disappeared.

This extends the previously verified 3M result but does not establish long-time
stability. The next diagnostic target is the puncture-cell chi/RHS update. No
new evolution or numerical repair was launched during this status check.
