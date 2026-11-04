# Coordinate-normalized inner shift: fresh boosted N16, N20, N24

User authorized sequential fresh N16, N20 and N24 evolutions to10M, all on four
Aurora debug-scaling nodes (48 ranks/tiles). Branch `codex/boosted-normalized-shift`.
Read state.json and remote boosted-normalized-shift-state.json before actions.
No duplicate builds/submissions. Existing BC2 baselines are terminal and preserved.

Current: **N16 S16 job9037573 submitted**, four nodes/48ranks, target10M. Plan
`campaigns/n16-boosted-normalized-shift-v1-s16-evolution-segment01/plan.json`,
SHA2561436287663dc79a9f40265aac0e1dd01f540ca2d4e55243aa3c0c750376174bd.
N20/N24 not yet submitted; advance sequentially after prior disposition.
Build attempt02 completed and froze; all launcher/science/build records are
locally preserved and verified in `aurora-build/`. Do not rebuild/freeze/smoke.
Source manifest SHA25615003c908c6c2640140045ed0473a93a8ffa9e19ab5dac868cd401358c5c42b5;
actual evolution SHA2568eb9c7ccd60603718d3c4c686af68cde784d70a2b1e9fb51c7ecaed3f8d9af08.
Binding footprint8.070GB, headroom1.236GB after output/atomic reserves. Both debug
queues checked before submission; see immutable records in `n16-submitted/`.
Hourly monitor `aurora-normalized-shift-n16-n20-n24` is active. Two obsolete BC2
N24 remote files were locally+remotely hashverified and retired with receipt;
local originals remain. No old failed checkpoint is a restart input.

## Single scientific change and its limits

On the inner plateau the initial map is x=kX,t=T, k=.306142283363079 for the
native C6 areal4M--12M slice, M=1/3. Determinant-one conformal metric components
are unchanged under this constant spatial rescaling, Lambda_x=Lambda_X/k,
beta_x=k beta_X. Thus the reference Cartesian driver

    (partial_T-beta_X.grad_X) beta_X = Lambda_X - eta beta_X

pulls back to

    (partial_t-beta_x.grad_x) beta_x = k^2 Lambda_x - eta beta_x.

Set inner Gamma coefficient from1 to k^2=.09372309766275977; retain eta=6.
The existing complete-RHS blend applies this change with inner weight s(r),
equal1 forr<=.2 and0 forr>=.56. The change is exactly
s(r)*(k^2-1)*Lambda. The characteristic-speed estimate uses the same coefficient.
Outer shift equation, restoring collar and compatible scri closure are unchanged.
This matches the reference driver only on the constant-map plateau, not the
entire Z4c discretization. FD4, block-local KO8 and chi guards remain unchanged.
The earlier normalization audit motivates this test but is not a failure-cause
or stability proof. Native map scale is independently checked by the oracle.

Separate targets end in `_matched_cfl_normalized_shift`, checkpoint BN1 (208-byte
scheme3). Old BC2 checkpoints cannot load; fresh native t0 at each resolution.
Scientific source680ce2f812036d6e5bc6009b6d64eaabe126d0cb. Immutable namespace
source/build/launcher-boosted-normalized-shift-v1 (build prefix build-sycl-mpi-).
Freeze actual binary hashes only after successful make-j64 build.

All other BC2 science is retained: exact boosted Schwarzschild v+.1, zero inner
initial shift, precollapsed lapse, C6 areal4M--12M slice, refined Cartesian core,
six shells, broad.6-.9 restorationcoefficient4, W.2-.56, KO8/originalscri,
auxiliaryD/lapse restoration/damping, directchi/lapse RK+physicalKO, floors and
pre-floor signs, angular2, nominalscri1, fixed effective penalty.6510416666660156.
N16dt.00025/core64^3, N20dt.0002/core80^3, N24dt1/6000/core96^3. M1/3,
coordinate endpoint10/3. Tracker and moving fastflow12/24/online tolerance.001,
diag.05. Post expansion quality threshold.01; preserve native acceptance and
missing masses, no radius-to-mass inference. No AMR or unrelated numerical repairs.

## Execution and monitor

Use /usr/bin/python3.11 remotely; SSH hzhu@aurora.alcf.anl.gov with ControlPath
/Users/hz0693/.ssh/codex-control/%C. Root /home/hzhu/hyperbolic-core-shell-20261028.
Reuse established launcher logic, four debug-scaling nodes/48ranks/12pernode,
CompactBinaryMerger, worker3200s/command3100s/native2920s (180s reserve).
User waived GPU smoke gates. A local nontrivial gauge oracle and two-step native
SERIAL check verify the new science; neither is a long/GPU stability result.

After build freeze mirror and hashverify all source/launcher/build/science
records. Helper plan --variant S16|S20|S24 --purpose evolution; bind unchanged
manifest, check ALL user's debug/debug-scaling jobs and saved submissions,
submit once. Healthy fully verified SAME-formulation/resolution walltime
checkpoints automatically continue sequential segments with checkpoint/hash/
horizonseed after matching terminal lock release. Numerical failures are
terminal; preserve/localize, never replay. Advance fresh N20 after N16 terminal
disposition, then fresh N24 after N20 terminal disposition; no automatic equation,
dt or floor adjustments. Diagnose infrastructure/integrity/storage first.

Home task cap10GB INCLUDING output/atomic/build reserves. Never weaken guard.
Retire only obsolete outputs after fresh local hash/length verification and
remote rehash, save receipt; preserve scientific snapshots and active restart
until verified successor. Remote full verification may precede local download,
but never purge before local preservation. Check exact BN1/config/n/dt/coefficient/
penalty/tracker/source/binary/input/artifact identities, finitepayload,time/steps/
progress and scheduler/worker/native disposition at every terminal segment.

Hourly quiet monitor: notify meaningful changes/failure/completion/required
input only. Keep local/remote state and monitor prompt current. Delete after
all requested dispositions, preservation, comparisons and report complete.

## Analysis

Compare with original BC2 N16/N20/N24 using their own helpers/identities; do not
rename failed BC2 endpoints as admitted states. Preserve full/own-candidate
constraint histories, moving cuts, trajectory/core margin, horizon quality and
mass gaps, scri null/pole/clock, segment-summed floor/sign counts with boundary
separate and runtime. Cumulative maxima are not instantaneous residuals.
Only admitted endpoints get matching BN1 diagnostics, common exterior cuts and
unsmoothed radial/field/gradient profiles. Do not invent fixed-cutoff histories.
Compare survival and accuracy separately; finite resolution trends and survival
alone do not establish fourth order or prove the stronger driver caused failure.
Inspect plots, commit/push reproducible evidence and conclusions.

## Build transport repair

Attempt01 failed before scientific targets compiled: macOS system tar generated
AppleDouble `._*.cpp` metadata entries, picked up by Kokkos source globs. The
manifest-bound scientific sources and launcher files were unchanged. Preserve
failed logs/build under `*-failed-attempt01`, quarantine only unmanifested
AppleDouble files (verify magic/hash/inventory), then explicitly retry make-j64.
`repair_transport.py` records this one-time repair and rejects a second use.
After quarantine the source-file inventory must equal the exact manifest.
No successful binary or frozen science was overwritten. For future transfers
use Python tarfile (no filesystem xattr metadata) rather than macOS system tar.
