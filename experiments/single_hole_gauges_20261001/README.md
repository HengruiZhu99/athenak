# Bounded single-hole gauge campaign (2026-10-01)

This is the user's single-black-hole-only follow-up. Binary collisions and Chau's
calibration remain manuscript placeholders. Do not submit to debug-scaling or
alter the unrelated SANE job there. Project: `CompactBinaryMerger`.

## Scientific matrix and existing evidence

`cases.json` orders 30 short cases: nine gamma=5 runs (three gauges × blocks
32/48/64, t=4M), then 21 stationary-puncture pulse/control runs (t=6M).
Gauges are telegrapher (tau=.1,kappa=.2), vanilla advective 1+log, and SSL
(amplitude .6, time 20, index 1). All share RK4, CFL .25, sixth-order space,
KO .1 and the same shift. Each resolution uses root_nx=4*block_nx.

The physical static mesh is exactly identical within each experiment, across
both resolution and gauge: 568 leaves, domain [-64,64]^3. Boosted refinement
level 6 covers x=[-1,2], y,z=[-1,1]; stationary level 5 surrounds the puncture,
level 4 covers the x-axis pulse region and level 3 surrounds it. Generated
mesh trees must match at all three resolutions, not merely region labels.
Check that each moving horizon remains adequately covered by fine cells.
Existing gamma=5 tel tracker is x≈.728 at t=4; other gauges require actual checks.

Earlier 20M BHaHAHA results at gamma=1,1.5,2,3,4,5 and fast-flow results at
1,1.5,2 are reused from `../telegrapher_aurora_20260929/results`. Existing
fast-flow gamma=5 L6/L7 chains are partial (2.039848M / 6.625960M), not 20M
qualifications. The earlier L7 comparison adds refinement and is NOT the present
fixed-geometry convergence test. A new short tel baseline is necessary because
old runs lack the 3D boundary-screened integrals and use moving AMR. Do not
repeat the completed boost ladder. BHaHAHA cold starts use identical angular
and radial settings at all resolutions. These runs study field resolution;
angular horizon convergence remains a separate question.

Stationary data are isotropic Schwarzschild with zero extrinsic curvature and
shift and pre-collapsed lapse psi^-2. Add a compact 3D bump centered at (4,0,0):
`delta_alpha=.2 exp[-q/(1-q)]` for `q=|x-center|^2/width^2<1`, zero otherwise.
The support radii are 1 and .5M at block32; the narrower pulse and zero-amplitude
controls repeat at block48 and block64. This is not a radial shell around the
hole. The pulse affects only lapse, not initial ADM geometry or constraints;
restarts do not reapply it. Controls remove the background gauge relaxation.
Compare inward/outward branches, peak amplitudes, width and maximum gradients;
use actual evolving metric and shift for coordinate characteristic speeds.
The 1+log estimate is -beta^x ± sqrt(2 alpha gamma^xx), not just sqrt(2 alpha).
Do not claim shock formation or a resolved ordering without resolution evidence.

## Convergence and boundary protection

History columns retain the original nine integral positions and append H²,M²,C²,
physical volume, boundary-excluded volume, sampled-speed-violation volume,
coordinate volume and chi-excluded volume. These are integrals; normalized
L2 is sqrt(integral/physical_volume). Norm regions are FIXED cubical shells:
8<=max(|x|,|y|,|z|)<16 (boost) and 2<=max(...)<8 (pulse). Near-hole spike
profiles are separate diagnostics, not a replacement for these volume norms.

Reject convergence if any of the three exclusion/violation columns is nonzero.
Require the coordinate volume to equal 28672 / 4032 respectively at every time
and resolution. A moving chi cutoff otherwise invalidates fixed-domain claims.
Cells closer than 4+8t to ANY domain face are excluded. The speed envelope 8
is deliberately much larger than expected gauge/light/shift speeds, with a 4M
buffer. A sampled audit checks physical inverse-metric light/lapse speeds and
conformal Gamma-driver estimates at history outputs. This is a conservative
empirical screen, not a proof of nonlinear hyperbolicity or a continuous-time
speed maximum. Discuss that limitation explicitly in the paper. The finite
windows remain outside boundary causal contact under this envelope.

Use synchronized physical times, common coordinate samples/masks and three
resolutions (nonconstant resolution ratios 1,1.5,2); do not blindly use log2
ratios for an order fit. Check signs and asymptotic behavior before reporting p.
Line output selects nearest transverse cell centers, which differ with dx;
account for this sampling when interpreting line-profile convergence.

## Reproducibility and bounded execution

Local CPU: `cmake --build build/local-serial -j 8`, then
`python3 experiments/single_hole_gauges_20261001/check_local.py`.
It tests lapse-only perturbation, identical initial constraints, diagnostic
vetoes, and restart equivalence. `check_controller.py` tests eight submission
guards. `generate.py` regenerates all inputs and their SHA256 identities.

Remote campaign root:
`/lus/flare/projects/CompactBinaryMerger/hzhu/telegrapher_lapse_20260929`.
New outputs/state: `single_hole_20261001/`. Executable:
`source/build/aurora-sycl-v5/src/athena`. Build log: `build_v5.log`.
The existing build script records source commit, binary hash, toolchain and
Kokkos compatibility patch. A hash marker is required before submission.

Run on Aurora from any directory:
`python3 source/experiments/single_hole_gauges_20261001/advance.py`
(with the absolute source path as needed). Inspection is the default. Add
`--submit` to submit exactly one next case; `--queue capacity` is an authorized
fallback after a confirmed debug rejection or occupancy. Prefer debug. Never
queue multiple campaign jobs or bypass PBS limits. One node for block32/48,
two nodes for block64; each reserves one hour, app stops after 50 minutes.
At most two attempts/case and 84 reserved node-hours total. No automatic
retry of a failed/partial evolution: inspect first and document any repair.
The finite manifest excludes new case families, binary runs and long evolutions.
If block64 performance requires a short four-node capacity retry, the user
authorizes necessary single-hole tests on capacity: document the evidence,
update and test the controller/manifest explicitly, and retain the finite
node-hour and attempt budgets. Do not mistake this initial allocation choice
for an extra user-approval requirement.

Controller uses a filesystem lock and persists submission intent BEFORE qsub.
An ambiguous submission stops all further submissions until PBS is reconciled.
An explicit queue-limit rejection is safe to retry on the next heartbeat.
It checks actual PBS completion, time reached, finite histories, horizon
success/residuals, input hashes and fixed safe volume before advancing.
`needs_review` is not success. Never blindly clear it. Preserve failed evidence,
fix a demonstrated issue within limits, record rationale, then explicitly
prepare another attempt if justified. Unique PBS IDs retain all attempts.
The controller does not itself schedule wakeups; the Codex heartbeat does.

Only compact histories, tracker/horizon diagnostics and x-axis profiles are
saved; profiles are gzip-compressed after completion. No periodic or final 3D
checkpoints in this new short matrix. Stop submitting at 5GiB new output.
Do not delete unique data to hide a failed result or get around this limit.

Legacy capacity jobs 8883678,8888791,8888792 are on user hold, imposed in this
turn to serialize this campaign. 8883678 is the older added-level BHaHAHA test;
8888791/2 are partial fast-flow gamma5 continuations. Leave held during the
new matrix. After the new analysis, cancel the obsolete unstarted added-level
BHaHAHA request, and either redirect the two useful partial FF continuations
to debug/capacity within a separately recorded short budget or leave them
explicitly deferred and report it. Do not silently release the old 8/16-node
requests or claim these held runs completed. Preserve their restart sources.

## Completion and publication handoff

After numerics pass, copy compact finalized evidence locally, retaining hashes,
job receipts, exact inputs and source/binary identities. Save scripts and
PDF/PNG figures: (1) fixed-shell H/M constraint convergence, (2) near-hole
constraint spikes with refinement faces and horizon position, (3) horizon
mass/residual comparison, (4) pulse-minus-control spacetime panels for three
gauges, (5) inward steepening/width/amplitude with resolution checks and
coordinate-characteristic overlays. Do not draw a propagation conclusion
from raw lapse alone. Keep legacy figures and their qualifications.

Update the existing manuscript repo `/Users/hz0693/research/gauge_exploration/telegrapher_gauge`
in place (`arxiv.tex`); preserve the already-written PRD literature revision.
Replace proposed radial-shell wording with the actual compact off-center bump,
add verified numerical outcomes and limitations, keep binary/calibration
placeholders. Compile via existing latexmk workflow and visually inspect the
PDF; native LaTeX compiler cannot access the companion figures in this project.
Use the presentations skill to create an editable research slide deck, with
figure provenance, clear gauge/resolution labels, and explicit open questions.
Inspect rendered slides. Commit and push code/results to
`HengruiZhu99/athenak:project/telegrapher_lapse` and manuscript/deck to the
existing `HengruiZhu99/telegrapher_gauge` branch after reviewing diffs. Never
change original PR #790/#792 branches. Pause the monitor only after final
handoff, or on an actionable block that prevents safe progress; report which.

## Monitor installed

Heartbeat `aurora-single-hole-gauge-comparisons` is active every 15 minutes in
this thread. It advances the guarded controller and executes the publication
handoff above, notifying only meaningful changes or actionable failures.
See `storage_cleanup.json` for 79.73GiB of reproducible initial checkpoints
removed; all later checkpoints, held-job restart sources and diagnostics remain.

## Initial submission

GPU v5 build completed successfully from source 933b69ee. At 2026-10-02
00:55 UTC the one-node debug request was explicitly rejected by PBS (code38):
`would exceed queue generic's per-user limit of jobs in 'Q' state`. No debug
job was created. The authorized capacity fallback accepted job **8890549**,
case `g5_tel_b32`, one node, one hour, account CompactBinaryMerger. It was
queued at the last setup check. Receipts and binary identity are in
`setup_evidence/`; live remote state takes precedence over this snapshot.
