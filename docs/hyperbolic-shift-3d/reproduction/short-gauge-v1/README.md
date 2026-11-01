# SG1: restore the earlier gauge interval, retain the direct outer driver

The N16 accuracy investigation continues after LG1 failed at 2.9715M. The next
controlled test uses the IT1 interpolation interval 0.2--0.56 with the DO1 direct
outer driver and the DG1 restored IT1 constraint damping. Only the gauge weight
interval changes relative to DG1. The physical inner telegrapher is exact for
r<=0.2; the direct outer branch applies for r>=0.56. Its scri limit and original
restoring collar are unchanged. The initial lapse/shift rates match IT1 when
B=D=Lambda=0; B rates and subsequent evolution need not match.

This tests whether retaining the Cauchy inner gauge deep into the initial
hyperboloidal slicing transition generates the harmful pulse. It does not assume
the interval is optimal or exclude a discretization contribution. The shorter
interval is steeper than DO1, but it is the one previously evolved by IT1.

Fresh original N16 wormhole text, dt=0.0005, target coordinate time 10/3=10M.
Direct chi/lapse RK and KO; physical lapse floor 1e-10, positive chi floor 1e-8
inside r<0.03, finite nonpositive chi rescue everywhere. Preserve grid, FD4, KO,
angular epsilon=2, scri penalty scale=1, horizon lmax12/ntheta24/tolerance0.001,
and diagnostic interval0.05. Count pre-floor sign candidates, preserve horizon
quality flags and missing values. N24 is not authorized until reviewed N16
accuracy success; completion alone is insufficient.

Unique checkpoint HYP3D_MPI_SG1, header168, scheme3. No older checkpoint may be
loaded. The N16-only controller/helper reject wrong schemas, failed/unadmitted
states and changed parameters. Healthy walltime continuation requires full
artifact SHA256/length verification, admitted configuration and finite payload,
report/header matching, frozen bindings and forward progress. Preserve horizon
seed sidecars. Never resume a numerical failure or silently change science.

Use one CompactBinaryMerger/debug node, 12 MPI ranks/GPU tiles, worker3200s,
command3100s, exit reserve180s. Check all user debug jobs and saved submissions
before submitting; never duplicate queued/running/ambiguous jobs. Keep home task
footprint below5,000,000,000bytes including output and atomic checkpoint reserves.
Copy/hash-verify evidence locally before remote purge. Release only the matching
terminal root lock. Build SYCL/MPI/PVC with make -j64.

After verified completion compare full/candidate-exterior histories and matched
native endpoint constraints against IT1, DO1 and DG1, using one common cutoff
outside all candidate horizons. Locate radial error bands, assess floor/sign
counts and horizon acceptance. On failure collect and localize it without
restarting. Do not claim convergence, global hyperbolicity or accuracy from
survival. Frozen build and submission records will be appended once available.

## Local verification and frozen source

Scientific source: `47ae0c35dcef877550ac4371e74d0948842b68b7`.
Nine gauge/device/characteristic checks passed, including the new prescribed
interval and unchanged damping oracle. Fresh two-step evolution to coordinate
0.001 and one-step restart to0.0015 both completed with SG1 schema; maximum
surface/ghost null residual stayed below1.2e-15. These are smoke checks only.
The first controller-suite invocation loaded a version missing SG1 in the
experimental executable allowlist. That was corrected before this source commit;
the focused campaign tests and the full rerun of all127 controller/campaign
tests passed.

`compare.py` requires verified healthy10M completion and matching native SG1
checkpoint diagnostics. It never accepts a failure checkpoint and never grants
N24 automatically. It records one common exterior cutoff, radial error bands,
segment-local sign/floor counts and horizon acceptance gaps. Syntax checking
is not endpoint validation; no SG1 long-time result is available yet.

## Submitted Aurora trial

Job **9018835** is submitted, plan
`campaigns/n16-short-gauge-fresh10m-v1-segment01/plan.json`, SHA256
`0b4eb0b2b886bf86c9cff3ea5f5a4b215f840158a82347d71907c410362d4f5f`.
Follow `short-gauge-10m-state.json` and saved submissions before any action.
Do not submit another copy. N24 remains gated on reviewed N16 accuracy success.

Frozen source manifest `source-short-gauge-v1-manifest.json`, SHA256
`a5fd4a1cf9924927dfa19ddf9d27e91c6d13a91a0d12b8e997e6868853801cff`.
Binary `build-sycl-mpi-short-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_short_gauge`,
SHA256 `1b75a50ec11f8b3908de9172bd5baec57eb8bc17d382d1fba25173ab4352996f`.
The frozen launcher is `launcher-short-gauge-v1`; verify all files against its
`provenance.json`, including `short-gauge-science.json`, before using
`control.py` or `short_gauge_campaign.py`.

All546 archived source files and1486 Kokkos files were hash-verified. SYCL/PVC+MPI
build completed with oneapi/release/2025.3.1 and make -j64. Binding measured
3,684,233,216bytes and1,088,481,201bytes remaining after planned/atomic reserves.
There were no user debug jobs before submission; unrelated capacity jobs remain
held. The new20-minute monitor is `aurora-n16-original-interval-gauge-to-10m`.
It may continue healthy walltime checkpoints only; delete it after verified
trial disposition. The persistent investigation remains active.
