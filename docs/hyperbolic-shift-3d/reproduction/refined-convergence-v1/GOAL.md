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

Current progress: compatible stationary outer reference gauge v2 is locally qualified and building in a separate immutable Aurora snapshot; it has not yet evolved. The earlier v1 reference precursor was caught before submission and is unused.  enlarged geometry/initial-map parameters and matched-plateau
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
