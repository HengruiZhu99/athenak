# Persistent goal: regular and convergent v=0.1 puncture through 10M

User authorization: investigate slicing versus numerical/interface origin of
nonconvergence; use hyperboloidal_refined, enlarge the Cartesian region and move
gauge transitions outward as appropriate, compare with telegrapher_lapse inner
Z4c, keep consistent conformal factors in geometry/initial data/evolution, test
stationary before boosted holes, vary resolution, and retain scri behavior.
Four Aurora debug-scaling nodes are authorized; up to eight if necessary with
several independent cases in parallel, subject to actual queue limits. Never
interpret survival or two resolution residual slopes as convergence proof.

Work branch codex/refined-convergence is based on hyperboloidal_refined
21676d39e, in /Users/hz0693/.codex/worktrees/refined-convergence/hyperboloidal.
The separate original worktree/branch codex/boosted-normalized-shift retains
its frozen science. All BN1 resolutions are complete, their artifacts verified
and three-resolution analysis pushed (e8eaa56d2; final state c41a5ef1a).
Its monitor was deleted after follow-up. Do not resubmit or repeat completed
BN1 diagnostics. The exterior residuals do not decrease under refinement.

Current progress: compatible stationary outer reference gauge v2 is locally qualified and frozen in a separate immutable Aurora snapshot; stationary N16 segment01 job9038410 ended at a fully admitted healthy walltime checkpoint4.782375M and segment02 job9038494 completed10M and all152 final artifacts are locally verified. All152 first-segment artifacts are locally verified. At this checkpoint99.868%/99.843% of exterior H²/M² lies in r.90–.95; matched evolved radial controls remain pending. Initial N12/N14/N16 exterior constraint residuals decrease with fitted orders3.89/3.66, but no evolved convergence is established. The earlier v1 reference precursor was caught before submission and is unused.  enlarged geometry/initial-map parameters and matched-plateau
controls implemented; local geometry/frame/oracle checks and MPI SMR restart
qualification passed. Production scheduled diagnostics/walltime support and
the frozen Aurora build are complete. Stationary6 N16 RF2 job9038055 completed10M natively with large exterior
errors. Original worker/PBS failure was executable permission, independently
verified with identical verifier bytes; all151 artifacts are preserved. The frame/separated-speed build completed. Its stationary20 job9038253 failed
at0.463875M; all151artifacts are preserved and its state must not restart.
Matched constant-driver job9038285 also failed at the same accepted step; both are fully preserved and neither may restart.
consult state.json for current disposition before any action. Initial halo controls isolate
a numerical frame-interpolation error. The principal audit motivates a
separate gauge control but does not establish the sole cause of late growth. User explicitly authorizes SMR/AMR as needed and measured performance
work; reuse existing infrastructure. Historical scientific source b41c24e8879 remains immutable. The new opt-in
controls freeze source342715a90 in a separate namespace. Analysis tools and
configuration identities distinguish both experiments.

Completion needs: geometry/frame consistency and spacelike/map regularity
checks; nontrivial inner equivalence with the actual reference Z4c; controlled
interface/derivative attribution; stationary then v=.1 production evolutions
through10M with complete dispositions and provenance; quantitative exterior
fixed-domain constraint and field-resolution trends; resolved field/derivative
features throughout evolution; scri null/pole/clock behavior; moving-horizon
acceptance/mass gaps, trajectory, floor/sign counts and cost. Explicitly report
remaining limitations and do not mark complete without evidence for all parts.

New radial isolation: initial native RHS residuals in .90–.95 decrease with resolution (fitted orders3.64–3.88); selective fourfold radial refinement reduces them206–246x. The corrected outer gauge survives beyond the old failure, but large evolved exterior residuals remain. See RADIAL-CONTROL-PLAN.md: fresh temporal control9038572 completed.6M and its admitted same-grid comparison shows exterior changes below.002%; paired radial control9038626 completed and is fully preserved. The full stationary/boosted convergence goal remains active.

Initial findings: RF1 narrow Omega join .18–.22 worsens short-time exterior
constraints. Its parser prevents a wide join because gauge onset must remain
inside the old affine map radius .297388. Larger useful inner regions require
consistent physical initial-map/slicing parameters, not bypassing that guard.
The physical spatial map and conformal frame are distinct. BN1's 10M error is
localized near first shell interface .305 and the gauge transition, with
coincident evolved-copy disagreement, not established slicing breakdown.
Evidence on original branch ab045abef under convergence-localization/ remains
immutable. New audits will distinguish initial spacelike margins, map
Jacobians, evolved metric regularity and numerical derivative sensitivity.

The admitted .6M radial control now reduces exterior H/M by36.394x/59.355x at the same timestep, while nominal Cartesian fields differ by at most4.24e-22 and floor/sign counters match. This establishes a major early spatial-discretization/interface contribution in .90–.95, not full10M convergence. GPU ownership benchmark9038692 completed: byte-identical endpoints, weighted partition0.94% slower. The candidate is rejected; retain equal production ownership. Next is stationary radial18 N12/N14/N16 to10M, with scaled timesteps and explicit reuse of the admitted N16 .6M state. Then qualify enlarged symmetric SMR support before boosted v=.1. No long-time convergence established yet.

Static symmetric-tube SMR is now a tested optional native policy (400 leaves): old policy unchanged, MPI parity and cross-rank restart arrays identical. It expands coverage, not finest resolution. The exact boosted initial data pass independent sampled fourth-order constraint differentiation and positive Q/Q0/map checks, but a directional stencil control identifies dominant outer **angular** truncation. Both angular and radial refinement recover near-fourth-order point residuals. A global boosted evolved convergence result remains pending. Stationary radial18 N12 job9038758 was observed through.6M; SSH expired before terminal reconciliation, renewal requested once. Do not resubmit or interpret unknown terminal status.

## Current priority: timestep and GPU-hour gate

The user explicitly confirmed `origin/project/telegrapher_lapse`
(`b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a`) as the Cartesian comparison.
Investigate multidimensional CFL estimates, actual limiting locations, source/
penalty restrictions and stable larger timesteps before the boosted long runs.
The requested gate is approximately fourfold **measured GPU-hour reduction**
with stability, not merely multiplying dt by four. Preserve comparable physical
duration, mesh, diagnostics and accuracy evidence; disclose equation changes
if adjusting damping or penalties. A norm-based speed is a candidate estimate,
not an automatic stability guarantee for mapped multidimensional operators.
If the controlled investigation cannot achieve the target stably, report the
evidence and block the goal rather than launch the three-resolution campaign.
No improvement is yet established. Current stationary9040850 finishes unchanged;
collect and preserve it, but defer another stationary allocation. The immutable
boosted-transition baseline build is running; do not duplicate it or submit a
boosted long evolution before this gate.

Latest user priority: qualify and benchmark24³ meshblocks/two SMR refinements with merged shells before larger-CFL tests. Native topology has254 blocks and3.75million evolved cells, same finest1/480 spacing; no measured speedup yet. See LARGE-BLOCK-CFL.md. The frozen boosted baseline build is complete and locally preserved; no boosted long run submitted.
