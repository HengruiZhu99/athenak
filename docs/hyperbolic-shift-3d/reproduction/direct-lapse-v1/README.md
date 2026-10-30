# Direct lapse retry

The chi-floor run failed at 9.156M with positive-infinite lapse in the eight
nearest puncture cells. This experiment tests the user's proposed direct-alpha
update instead of logarithmic lapse RK evolution. It retains the existing
telegrapher PDE, log-chi RK/KO, chi floor1e-8 at r<.03, scri closure, interpolation,
dt=.0005 (dt/dx=.0444444), n16, RK4 and geometry/projection policy. The lapse
perturbation alpha-A (static A=(1+r²)/2) now advances directly at each RK stage
and final compensated update; finite candidates are floored at physical
alpha>=1e-10, equivalently conformal alpha>=1e-10*Omega. Compensation is cleared
when a final lapse update is clipped. Nonfinite candidates are still rejected.
Lapse KO now acts on the stored linear lapse perturbation. Positive interpolation
and unused logarithmic lapse chart construction remain as in the control.
This is a combined RK-variable/KO-variable comparison, not an isolated RK-only
change. No claim of long-time stability or equivalence to Cauchy Z4c is made.

## Main branch audit

Fetched HengruiZhu99/athenak main at c8d5c33e39089f005e416bf839652f6796fa6e82.
In src/z4c/z4c_update.cpp, ExpRKUpdate linearly combines stored fields and their
RHS at each explicit RK stage. src/z4c/z4c_calcrhs.cpp lines535-547 computes the
lapse RHS using advective 1+log/harmonic slicing and optional slow-start lapse;
lines570-582 applies KO to all stored variables. No lapse floor was found in
main's src/z4c. The optional floor in z4c_tasks.cpp lines362-384 applies to chi;
it is disabled by default in z4c.cpp. Therefore direct lapse evolution agrees
with main's variable choice, but our lapse floor is the retained donor policy,
and our telegrapher gauge is not main's slicing equation. Similar gravitational
bulk equations alone do not establish the same puncture stability.

## Checkpoint and campaign policy

The new executable is hyperbolic_shift_3d_mpi_direct_lapse. AL1 checkpoint magic
HYP3D_MPI_AL1 has a168-byte header; PositiveConfig.scheme=2 at byte144 identifies
direct lapse RK/KO, with chi floor1e-8/radius.03 at byte152. Existing schemes
retain their original identities and default behavior. The explicit importer
accepts only the verified admitted DT1 checkpoint at3M with SHA256
628c2109bdf7ca6624afff58b65e8a28baca724e9e73de17b517454b26fe6a31.
Physical fields and non-lapse compensation are unchanged; volume lapse
compensation is explicitly reset because its old units belong to log(alpha).
No failed checkpoint is admitted or reused.

Aurora target is20M from the verified3M state with n16, dt=.0005, original
penalty strengths, diagnostic interval.05, horizon lmax12/ntheta24/tolerance.001.
Use CompactBinaryMerger/debug, one node at a time,12 MPI ranks/GPU tiles,
SYCL+MPI, make -j64, and home footprint below5,000,000,000 bytes. Freeze a fresh
source/build and bind hashes before submission. Continue only healthy walltime
checkpoints; numerical failures must stop and be reported. Large remote evidence
must be copied and SHA256-verified locally before purge.

## Local validation

All53 controller tests and log/direct floor unit tests passed. Two-rank n16
local tests completed a puncture step from3M and a subsequent restart, rejected
a CF1 checkpoint in the AL1 executable, and preserved Minkowski for two steps
with maximum field drift6.5106e-15 and H_L2=2.2791e-15. The local smoke runs used reduced horizon lmax1/ntheta4 for cost; the Aurora
comparison retains lmax12/ntheta24. These are short smoke
checks, not a long-time Minkowski or puncture result.

A split versus uninterrupted two-step comparison is bitwise identical, including
RK compensation, when the step sizes are identical. The first comparison using
exact decimal end-time targets differed by up to5.5e-15 in volume fields because
the driver's existing final-step clipping gave slightly different floating-point
step sizes; using nextafter endpoints to keep full dt removed that difference.
This endpoint behavior was not changed. The first-step lapse difference from
the log-lapse/KO control is2.43269e-5; bitwise equality between these different
numerical schemes is not expected.

Frozen scientific source:145515fcee36132bb6a1bc7e1ea095ffb8256f48.
Remote root:/home/hzhu/hyperbolic-core-shell-20261028.
Source:source-direct-lapse-v1; manifest:source-direct-lapse-v1-manifest.json.
Controller:source-direct-lapse-v1/tst/hyperbolic_shift_3d/aurora/control.py.
Binary:build-sycl-mpi-direct-lapse-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_direct_lapse.
First plan:campaigns/n16-direct-lapse-to20m-v1-segment01/plan.json.
Initial AL1 checkpoint SHA256:
7187756d749abedd0f3c6d2e4f1d731ba21e5ad6e7366b8cb80699e5a099c2f3.

## Automatic continuation protocol

After a terminal healthy walltime stop, validate the admitted AL1 checkpoint
against the native report and worker artifact hash: n16,nf31,kind2,admitted1,
dt.0005,time in[1,20/3),header168 bytes,length168+16*(31*cells+surface),
PositiveConfig.scheme2 at144,chi floor1e-8/radius.03 at152. Preserve all source,
executable and coefficient identities. Carry a horizon-seed sidecar only if
bound to that exact checkpoint; retain candidate-quality flags. Sum per-segment
floor counts rather than interpreting them as cumulative across restarts.

Use collect --release-terminal-lock on the matching terminal plan, check all
user debug jobs, create a fresh sequential segment plan with policy
n16_direct_lapse_retry_to_20M, bind, then submit. Do not resume numerical or
storage failures, unadmitted checkpoints or ambiguous scheduler outcomes.
On numerical failure or verified20M completion collect evidence, commit/push
results, report and stop the monitor. Keep verified-copy-before-purge and5GB
storage guards. If SSH expires request renewal rather than repeatedly submitting.

## Submission

SYCL/MPI build passed with make -j64 in173.16 seconds. Job8990333 was submitted
on one CompactBinaryMerger debug node with12 MPI ranks/GPU tiles and a one-hour
allocation. The30-minute monitor aurora-n16-direct-lapse-to20m is ACTIVE with
healthy-walltime continuation and stop-on-numerical-failure policy. Home campaign
footprint at binding was4,319,555,584 bytes, below5GB. See build-status.json,
binding.json, submission.json and monitor.json for exact provenance and policy.
The result is reported below.

## Result: direct lapse trial failed at9.0975M

Job8990333 stopped with invalid accepted volume geometry at coordinate t3.0325,
9.0975M, and checkpoint_admitted=false. The rejected state contains positive
infinite chi at global cells15616,28704,29184, near radius.02923. Their positions
and finite last-accepted chi are recorded in segment01/verification.json.
No other nonfinite volume field or nonpositive conformal-metric eigenvalue was
found. There were216 stage and101 final chi-floor interventions and8 lapse-floor
interventions. The old floor counts do not determine how many candidates were
negative, since positive values below the floor are counted too.

The recorded last RHS maximum was1.19038e120. Maximum null residuals remained
1.22126e-15(surface) and1.55431e-15(ghost). The last ordinary diagnostic was at9M;
its minima must not be reported as failure-state extrema. This was numerical
failure, not walltime or a finite-negative stopping condition. The failed state
was not used as a restart. Raw artifacts were copied locally and SHA256-verified
before remote checkpoint/failed-stage purge; the matching terminal lock was
released. A separate diagnostics-only replay, job8990381, starts from the same
verified3M AL1 input with pre-floor sign logging. See ../candidate-signs-v1.
