# Coordinate-normalized inner shift: fresh boosted N16, N20, N24

User authorized sequential fresh N16, N20 and N24 evolutions to10M, all on four
Aurora debug-scaling nodes (48 ranks/tiles). Branch `codex/boosted-normalized-shift`.
Read state.json and remote boosted-normalized-shift-state.json before actions.
No duplicate builds/submissions. Existing BC2 baselines are terminal and preserved.

Current: **N16 S16 job9037573 completed and verified at10M/13334steps**.
All five worker artifacts are locally hash/length verified; the admitted BN1
checkpoint SHA256 is8e8bfe0fd44f181ac06f62d27e96f6c6349536103b68540d92d289f75e5aa5e8
(259205840bytes). Native/worker/PBS all succeeded; matching terminal lock released.
Native command runtime2195.923s (36.60min); PBS reported41min28s on four nodes.
No N16 continuation, rebuild, freeze or smoke is needed.

**N20 segment01 job9037706 reached9.3174M/15529steps and checkpointed at
healthy walltime.** Frozen helper admission passed, all five worker artifacts
are locally hash/length verified, native/worker checkpoint disposition agrees
with expected PBS exit3, and its matching terminal lock was released.
Checkpoint481985232bytes, SHA256
f9e92791038645ca88cfcd8723209e1f71cfb809383199d5b714409a98d7ade5.
Native command2913.925s; this is not a numerical failure or completed10M result.
The frozen worker emitted no horizon-seed sidecar for this segment; continuation
binds the unchanged science and exact checkpoint, without inventing a seed.
Keep the changed finder initialization/sampling boundary explicit in analysis.

**Same-case segment02 job9037854 is submitted** under the unchanged
four-node/48-rank policy, dt.0002, target10M. Read `state.json` and saved submission
for its exact job. No duplicate plan or restart from another formulation.
Storage initially blocked binding. Two obsolete BC2 N16/unboosted N24 segment01
checkpoint copies were freshly hash/length verified locally, remotely rehashed,
and retired with receipt `retired-before-n20-segment02.json`; local originals
remain. Active BN1 N20 restart and scientific snapshots are retained. Successful
binding footprint7,919,046,656bytes and headroom807,791,740bytes after reserves.
N24 remains unsubmitted and follows N20 terminal disposition/preservation.

The monitor checks every30minutes. Evidence in `n16-completed/`,
`n16-local-verification.json`, `n20-submitted/`, and `n16-comparison/`.
`compare_n16.py` preserves separate BC2/BN1 scientific identities and moving
candidate cuts. BN1 has29/68 online accepted rows (last4.5M) and67/68 post1e-2
expansion passes, versus BC225/68 (last3.75M) and65/68. Final BN1 candidate
RMS.00532828 passes post1e-2 but the final mass is missing. Chi floor stage/final
counts fall from22188/7484 to5133/1723; lapse floor interventions fall3032to0.
The cumulative pole/clock maxima remain essentially6.03363/14.01299, and null
residuals stay at roundoff. These observations do not prove improved accuracy.

At10M the own-candidate H/M norms are.1173905/.1510484 for BN1 versus
.0273546/.0394122 for BC2, but the cutoffs differ (.15239 versus.19253).
Use `compare_endpoint.py` for the admitted endpoint comparison at common r>.3,
which exceeds all recorded complete candidate enclosing radii in both runs.
This is a common coordinate-domain comparison, not an invariant physical-domain
comparison or final-horizon acceptance. Missing masses remain gaps.

Completed common-cutoff r>.3 reconstruction reproduces full-domain report
norms and preserves the checkpoint hash. BN1 H/M/Mr=.03593065627/.05473442898/
.07097847590 versus BC2 .01448145953/.02303339626/.01378313985: ratios
2.4811/2.3763/5.1497. Thus the N16 weaker driver does not improve these exterior
endpoint residuals despite reducing floor interventions and delaying loss of
online horizon acceptance. Regional integrals and the inspected radial plot are
in `n16-comparison/`; native diagnostic operators are unchanged. N20/N24 remain
necessary to assess this family's resolution dependence; no convergence claim.


Build attempt02 completed/froze previously; all build/provenance records remain
in `aurora-build/`. No successful build is repeated. Previously retired BC2
N24 checkpoint/capture files remain preserved locally; do not retire them again.

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

Quiet 30-minute monitor: notify meaningful changes/failure/completion/required
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
