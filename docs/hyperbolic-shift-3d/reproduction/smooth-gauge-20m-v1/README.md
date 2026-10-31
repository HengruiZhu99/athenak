# Smoother gauge continuation: N16 then N24 to 20M

The user authorized continuing both verified smooth-profile 3M checkpoints to
20M, N16 first and then N24, with automatic resubmission after healthy walltime
stops and termination of each resolution on numerical failure. This supersedes
the earlier experiment's 3M stopping limit; it does not change its results.

Scientific source remains b27f76f37, manifest source-smooth-gauge-v1-manifest.json
SHA256 7308e7e5c3f54d960c0b4bae9b6cf92127a9aa2d66d7f46a5c88c0baaad64fdb.
Executable build-sycl-mpi-smooth-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_smooth_gauge
SHA256 53eda0563a1b48846ac7372422906c35569f6ed685b75bb1dc980ccea83567f1.
No recompilation or scientific source modification is needed. The separate
launcher-smooth-gauge-20m-v1 freezes only the updated scheduler/controller and
verification helper. Its provenance.json records commit and per-file hashes.

Keep GW1 log-chi/log-lapse evolution throughout, including after3M. There is no
conversion to the earlier direct-variable AC1 scheme. Keep lapse floor1e-10,
no chi floor, smoothstep9 weight on0.56–0.8, original harmonic collar, angular
coefficient2 and scri penalty scale1. N16 dt=.0005; N24 dt=1/3000;
both dt/dx=.0444444. Target coordinate time20/3 equals20M. Preserve horizon
lmax12/ntheta24/tolerance.001 and diagnostic interval.05, with complete-candidate
exterior diagnostics and unchanged horizon acceptance flags. Do not adjust any
of these settings after failure or between segments.

Verified start files, restored without changing any physical or RK data:
- N16: inputs/n16-smooth20m-at3m.checkpoint, 79725208 bytes,
  SHA256 35dcab50a93551a9db98e1ac71dd4034f0730f396fd06ea1ed7d01cca60aa7d4.
- N24: inputs/n24-smooth20m-at3m.checkpoint, 234413720 bytes,
  SHA256 409edeabf401d76426d9010fb2175c4141cfbad992d98c2f226dd5f142b10545.

Use one CompactBinaryMerger/debug node,12 MPI ranks/GPU tiles, worker3200s,
command3100s and native exit reserve180s. Check ALL user debug jobs before
submission and preserve unrelated jobs. Keep the task footprint below5GB;
reserve output/failure capture and an atomic checkpoint copy. Account for observed
filesystem block-count variation. Copy and SHA256/length verify large files
locally before remote purge; retain the current restart until its successor has
verified output. No large time-dependent 3D dumps are requested.

## Segment protocol

Read remote smooth-gauge-20m-state.json and the current plan's submission and
worker records first; never duplicate queued/running/ambiguous submissions.
Use launcher-smooth-gauge-20m-v1/control.py for bind/submit/collect and
launcher-smooth-gauge-20m-v1/smooth_campaign.py for plan/verify. Always bind the
unchanged scientific manifest above. Before use, compare launcher files against
provenance.json and check the scientific binary/manifest hashes.

The plan helper requires n16 or n24, a fresh segment number, an admitted GW1
restart and its independently verified SHA256. It checks the152-byte header,
exact dimensions/configuration/dt, finite payload and positive accepted chi;
optional horizon-seed sidecars are validated and bound. Example:

    python3.11 launcher-smooth-gauge-20m-v1/smooth_campaign.py plan --root "$PWD" --n 16 --segment 1 --restart inputs/n16-smooth20m-at3m.checkpoint --sha256 35dcab50a93551a9db98e1ac71dd4034f0730f396fd06ea1ed7d01cca60aa7d4

After a terminal job, collect via control.py, copy the full campaign directory
locally, and verify all worker artifacts' hashes/lengths before cleanup. For a
healthy completion/walltime stop, run smooth_campaign.py verify --root ROOT
--plan PLAN: it requires terminal scheduler, healthy worker, frozen bindings,
all artifact hashes, admitted GW1 configuration, finite checkpoint, matching
report time/steps/dt and forward progress. A completion must be exactly20/3.
Do not infer checkpoint admission from file existence or process exit alone.

After verification, release ONLY the matching terminal allocation lock (check
allocation-released.json first), update the state, and create a new sequential
plan with the verified checkpoint/hash if below20M. Bind, check ALL user debug
jobs, then submit once. Preserve horizon-seed sidecars when available. The
controller's healthy checkpoint classification permits only the expected
completion-status check to differ; other scientific checks must pass.

On numerical failure, collect/hash-verify evidence, mark that resolution failed,
release its terminal lock, and proceed to the next requested resolution from
its own verified3M checkpoint. Never resume a failed/unadmitted state or change
equations/floors/dt automatically. Scheduler/storage/integrity failures require
diagnosis before any retry. After N16 completes or numerically fails, start N24.
After both resolutions complete or numerically fail, collect available histories,
compare with the matched original/smooth3M evidence and original long campaign,
commit/push concise evidence/plots, report limitations and delete the hourly
monitor. Reaching20M alone does not establish accuracy or convergence.

If SSH expires, request renewal once and leave the monitor active; do not make
repeated submission attempts. Stay quiet for unchanged/non-actionable states;
notify on meaningful progress, continuation, completion, failure or required action.

## Initial submission

N16 job9009718 is submitted under the new continuation policy. Plan:
campaigns/n16-smooth-gauge-to20m-v1-segment01/plan.json, SHA256
4b1b315ec4fe06cdd3a02335718037daeccf3f5ca6687607c2d7a4ea037daba2.
N24's admitted3M input is restored/hash-verified and its segment01 plan is
prepared but unbound/unsubmitted, waiting for N16 completion or numerical failure.
The submission-time all-user debug query was empty. Binding measured3.154GB,
leaving1.618GB headroom after N16 output/atomic reservations; N24 upload completed
subsequently and is included in later accounting.

Frozen launcher commit: d5d82db30213942327184ed1078b81d1cd689d0f.
Controller/preflight/resolution suite:114 tests passed. Final focused controller
and GW1 checkpoint tests:57 passed, including rejection of changed target/dt,
wrong-variable executable, unadmitted/wrong-schema/wrong-config/nonfinite input
and preserving healthy walltime classification while rejecting other failed checks.
Both actual starting checkpoints passed the same GW1 validator.

Hourly heartbeat monitor: aurora-smooth-gauge-n16-then-n24-to-20m, ACTIVE at
submission. State and small initial submission records are in submission/.

## N16 terminal failure; N24 submitted

N16 job9009718 failed numerically at last accepted9.1575M (last diagnostics9.15M):
nonfinite bulk RHS at active cell3071, reason7. It was not a walltime or storage
stop, and the saved checkpoint is unadmitted. See n16-failure/ for verified
artifacts, failure location and diagnostic qualifications. Large failed outputs
were locally hash/length verified before remote purge; terminal lock released.
No N16 numerical retry is authorized.

N24 job9013517 is now submitted from its own verified3M GW1 checkpoint with all
scientific settings unchanged. Plan SHA256
b802985d061c5ab6855805270ec6d656bb7b8f9e379f9bf7046518641022339d.
The all-user debug query was empty before submission. The hourly monitor now
tracks N24 and must not resume the failed N16 state.
