# Boosted refined core: N20 then N24 on four Aurora nodes

User authorized fresh N20 to 10M followed by fresh N24 to 10M on four nodes.
Branch `codex/boosted-refined-core-resolution`. Read `state.json`, remote
`boosted-refined-core-resolution-state.json`, and saved submissions first.
No duplicate submissions. Completed N16 BC2 job9036411 remains the baseline;
do not rerun it or revive failed BG1 / superseded BC1 experiments. No AMR here.

N20 job **9037163** FAILED numerically at last accepted6.2082M/10347steps, plan
`campaigns/n20-boosted-refined-core-resolution-v1-c20-evolution-segment01/plan.json`,
SHA2567993b031f3318e97a1578c8dbfaa71fab5fb637ca3c287f5867b84ec1d8ac3a0.
Both user debug queues were empty before submission; unrelated capacity jobs
were preserved. Binding footprint7,466,598,296bytes and remaining headroom
1,260,240,100bytes after output/atomic reserves. No purge was needed. New
launcher commit a73da2f129301355c061f6973d1fc63383afb6e3; all science is reused.
All six artifacts are locally hash/length verified. See `n20-failure/README.md`.
The failed BC2 checkpoint is unadmitted and MUST NOT resume. Terminal root lock
released. Its obsolete remote checkpoint copy was retired only after local
and fresh remote hash/length verification; local original retained, receipt
`retired-n20-before-n24.json` (481992704 allocated bytes freed).

Fresh N24 job **9037378** is SUBMITTED (last scheduler check Q), plan
`campaigns/n24-boosted-refined-core-resolution-v1-c24-evolution-segment01/plan.json`,
SHA256ac42833394dbc8f3995807a64e2607c444f5db6888ced79e230f238a27d89427.
Four nodes/48ranks, native t0, dt1/6000, target coordinate10/3. Same frozen
science/source/binary; no rebuild/smoke. Both debug queues empty before submit.
Binding footprint7683278509bytes, remaining200883663bytes after output/atomic
reserves. Preserve current N24 checkpoint until a verified successor. Future
continuation may need retirement of other locally/remote hashverified obsolete
outputs; never weaken guard. Hourly monitor
`aurora-boosted-refined-core-n20-then-n24` remains ACTIVE; no duplicate N24 run.

## Science and distribution

Reuse frozen BC2 scientific source cb0f47b520afe4f88c98e11569f3ba88e632df9a
and `build-sycl-mpi-boosted-refined-core-cfl-v1`; no rebuild/freeze/smoke.
Manifest `source-boosted-refined-core-cfl-v1-manifest.json` SHA256
6d1c29a69bcce98d761d1f896b9f230553d7e4852d8a4c86384f33237634847b.
Science intentionally remains
`launcher-boosted-refined-core-cfl-v1/native-transition-science.json`, SHA256
b0124b10763a01abc9e1feb0436b6022078f320ad6c53ec40e989231cfdab17c.
Actual evolution binary SHA256
fbb8f8d2e9522d1b72fe28ea8b1e857f2b840f4d58365c5042a16727831f59e6;
matching diagnostic SHA256
525f2c2e0f45e138c6879c1d139cec15c2a381e799f2a049f9954b635fd36451.
New immutable launcher `launcher-boosted-refined-core-resolution-v1` binds
48 ranks, four hosts with 12 local ranks/GPU tiles each, and C20/C24 variants.
Verify its files against `provenance.json` before use. Never mutate old launchers.

The fixed topology has 64 Cartesian blocks and 36 shell patches (six layers).
The actual MPI ownership is equal contiguous active-cell ranges, not indivisible
blocks: cuts[r]=total*r/size, and scri rows have their own equal partition.
`partition-audit.json` binds this observation to the frozen scientific source.
Every one of 48 ranks has volume cells and scri rows. Equal cell counts alone
do not demonstrate equal computational cost or four-node scaling efficiency.

| Variant | Core nominal cells | Total evolved cells | dt | Checkpoint bytes |
|---|---:|---:|---:|---:|
| C20 | 80^3 | 969344 | .0002 | 481985232 |
| C24 | 96^3 | 1621728 | 1/6000 | 806091472 |

BC2 208-byte/scheme3 headers bind resolution and dt. Fresh native t0 at each
resolution, M=1/3, target coordinate10/3. No cross-resolution restart. Retain
exact v=+.1 boosted Schwarzschild geometry, zero inner initial shift/free gauge,
precollapsed lapse, native C6 areal4M-12M slice, broad .6-.9 restoring collar
coefficient4 and compatible original scri limit, W.2-.56, KO8/originalscri,
original damping/auxiliaryD/lapse restoration, direct signed chi/lapse RK and
physical KO. Fixed effective scri penalty .6510416666660156, actual penalty
CFL .25, nominal unlimited .384. Floors lapse1e-10, positivechi1e-8 at
origin-centered r<.03 plus global finite nonpositive rescue and pre-floor
counts. Tracker and moving fast-flow lmax12/ntheta24/tolerance.001, diagnostics
.05, angular2/scri nominal1. No automatic dt/floor/equation changes.

## Execution and automatic continuation

Remote root `/home/hzhu/hyperbolic-core-shell-20261028`, SSH
`hzhu@aurora.alcf.anl.gov`, ControlPath
`/Users/hz0693/.ssh/codex-control/%C`, explicit `/usr/bin/python3.11`.
Queue debug-scaling, account CompactBinaryMerger, four nodes /48 ranks.
Worker3200s, command3100s, native2920s (180s exit reserve), PBS one hour.
Authorized home-task cap10GB INCLUDING planned output and atomic reserves.
Check ALL user debug/debug-scaling jobs and saved submissions before each
single submit; preserve unrelated held capacity jobs.

Use the new frozen helper:
`native_transition_campaign.py plan --root ROOT --variant C20|C24 --purpose evolution`.
Bind via `control.py bind --root ROOT --plan PLAN --source-manifest source-boosted-refined-core-cfl-v1-manifest.json`;
then `control.py submit --root ROOT --plan PLAN` once.
At terminal, `control.py collect`, then helper `verify --root ROOT --plan PLAN`.
Require exact BC2/n/dt/config, source/binary/input/artifact identities, finite
payload, tracker agreement, forward progress, fixed penalty and consistent
scheduler/worker/native disposition. PBS3 with native/worker checkpointed
can be healthy walltime; numerical failures are terminal and never replayed.

After verified healthy walltime below10M, release only the matching terminal
root lock with `control.py collect --release-terminal-lock`, check the saved
`allocation-released.json`, generate the next SAME-variant segment with
`--segment NEXT --restart VERIFIED_CP --sha256 VERIFIED_HASH`; bind the horizon
seed, unchanged manifest/science, check queues and submit once. Remote full
verification may precede local download, but never purge without local exact
hash/length preservation and a fresh remote rehash receipt. Preserve the
current restart until its successor is verified. Never delete scientific
snapshots or unverified evidence; do not weaken the storage guard. Diagnose
infrastructure, integrity, storage or ambiguous submission outcomes first.

After N20 completes10M or a terminal numerical failure has been preserved and
localized, advance the independently authorized fresh N24 case. Never resume
the failed state. After N24 disposition finish comparison and preservation.
Maintain a quiet hourly monitor, with automatic healthy continuations and
sequential advancement. Notify meaningful progress, failures, completion or
required input; stay quiet on unchanged/nonactionable state. Ask for SSH
renewal once per outage. Delete monitor after both runs and follow-up complete.

## Analysis

**User's updated post-processing tolerance is expansion RMS <=1e-2.**
Use `horizon-post-policy.json` and `post_horizon.py`; preserve native1e-3 flags
separately. Do not change online finder arguments, restart seeds, frozen
plans/binaries or verification. Require a complete sampled candidate and finite
positive radius. This post expansion-quality selection is not proof that the
unrecorded native mass-stationarity condition passed. Rejected candidates have
NaN masses/areas in native summaries; preserve these gaps. Do not manufacture
mass from coordinate radius. An admitted endpoint may be re-evaluated offline
at1e-2 with the other finder settings retained. Existing constraint histories
and integration cutoffs are unchanged by relabeling candidate quality.

Preserve all artifacts, then compare to completed N16 BC2 using its old
two-node helper/science identity. Preserve trajectory/core margin, accepted
horizon mass and acceptance gaps, full/own-candidate constraints, scri
null/pole/clock, signs/floors summed across segments with boundary counts
separate, and cost. Only admitted endpoints get matching native diagnostics
with a common cutoff exceeding relevant recorded candidate enclosing radii;
do not infer a true horizon from an unaccepted surface. Preserve unsmoothed
near-puncture/transition field, gradient and constraint profiles and radial
errors. No fabricated fixed-cutoff histories. Record actual rank phase timings;
changed resolution and node count do not constitute a strong-scaling test.
N16 survived10M but had substantial floors and no accepted horizon after3.75M.
Space and time refine together; survival is not accurate evolution or a
fourth-order convergence proof. Inspect plots and commit/push evidence.
