# Goal 3: Schwarzschild preservation and wormhole-to-trumpet evolution

2026-10-23 exception: an exploratory native single-puncture pilot is explicitly
authorized before Goal 2, with fixed dt=1e-4 and stability prioritized over
accuracy and convergence. See the [pilot record](../validation/puncture-pilot-20261023/README.md).
The prerequisite and full completion gates below remain applicable to full
Goal 3 acceptance. Goal 2 is inactive and Goal 1 remains incomplete. No stable
puncture, stationary solution, trumpet, or exact-scri result is yet claimed.

The pilot's completion target is actual bounded gauge relaxation with resolved
lapse, shift, conformal factor and inner geometry, rather than qualification
alone. The current authorization ends at t/M=1; extending it requires a concrete
review and new user authorization. See the [continuation state](../validation/puncture-pilot-20261023/NEXT.md)
for the current separately versioned candidate, jobs and preserved failures.

Pilot update (2026-10-23): native balanced candidate v4 passed CPU/GPU/MPI/restart
qualification. Job 8958337 completed both meshes through t/M=.01 with all
predeclared criteria passing; geometry remains wormhole-like, with no trumpet
evidence and mixed outer-region resolution trends. Job8958401 subsequently
reached t/M=.028 on both meshes but stopped on the paired outer-region
resolution-growth criterion. The t/M=.05 milestone remains unachieved; a
passive localization audit and a qualified boundary-order comparison have
been reviewed. The boundary change is rejected; job8958489 reached t/M=.036
on both original meshes, then stopped on new layer momentum and connection-
symmetry resolution holds. Individual criteria and additional growth caps passed;
the raw paired criterion remains failed. Subsequent job8958533 hit fixed
outer-collar momentum caps at coarse t/M=.040, with fine at.036. Unchanged
evolution is held. Job8958574 rejected the proposed gauge feedback after its
frozen response-improvement gate failed; no feedback was installed. A separately
versioned initial-shift second-jet correction (v5) passed analytic and full
native qualification; job8958641 completed both meshes through t/M=.01.
Interface H is larger than v4, and no trumpet is seen. Job8958720 stopped at paired t/M=.016 on outer-collar H resolution growth.
The last two radial rows dominate; v5 remains held without waivers or cap
rebasing. A separately versioned radial boundary-accuracy correction is being
tested as v6 in job8958752. CPU both meshes, MPI, restart, passive monitoring
and independent diagnostics pass. GPU qualification also passed and actual
v6 reached paired.016 and failed the same collar-H growth criterion.
Its fine H/M errors are worse than v5 at that point; the radial-order
correction is rejected for continuation, with all results preserved. The separately
versioned v7 preferred-source/normal-feedback candidate passed its full source
admission in job8958870. Disabled-v4 CPU regression is bitwise; enabled coarse
CPU four-step checks, passive-monitor comparison and restart comparison pass.
Seven-rank CPU agreement and empty block sets also pass. Fine CPU and the
independent CPU diagnostic review pass. GPU qualification now also passes on
both meshes, including MPI, restart, passive diagnostics and actual tile binding.
Actual v7 evolution in the same allocation has passed paired .01 and .016,
including the targeted collar H/M and boundary gauge improvement test that
earlier candidates failed. The paired campaign stopped at .020 on an
outer-collar tangential-Lambda resolution-growth hold; per-mesh criteria still
pass and collar H/M/Z decrease on fine. Job8958870 is terminal; passive
decomposition job8958971 investigates this concrete hold before any continuation.
Inner geometry through .020 remains wormhole-like.
Its greater frozen transient amplification is recorded, not hidden by the
negative nonzero pole result. This is progress
within the exception, not full Goal 3 activation or acceptance.

Status: provisional; refine from Goal 2 before activation. Former milestone 6.
Prerequisite: [Goal 2](02-angular-minkowski.md) passes. Follow
[shared rules](README.md).

## Objective

Demonstrate convergent full-Z4c Schwarzschild evolution, first preserving a
stationary trumpet, then resolving the inner wormhole-to-trumpet transition,
while retaining the separate immutable Minkowski hyperboloidal reference.

## Execution

1. Admit the black-hole geometry and its constraints independently. Derive/check
   the actual hyperboloidal asymptotics and exact-scri limits for this chart;
   the scalar trumpet's logarithmic height coefficient is chart dependent and
   cannot be copied blindly to a different slicing.
2. Match stationary data to the chosen evolution gauge and its sources. An exact
   stationary four-metric is not automatically stationary under the selected
   moving-puncture gauge. Either establish compatibility or label the run as
   gauge relaxation; do not subtract its RHS to claim preservation.
3. Test stationary preservation using raw RHS checks and a three-grid evolution.
   Measure physical mass, null/regularity conditions and H/M/Theta/Z in the core,
   near the horizon, across the interfaces and at scri. Use horizon diagnostics
   where validated; do not confuse a coordinate sphere with a detected horizon.
4. Run the wormhole-to-trumpet transition separately, first a low-cost spherical
   diagnostic then the native 3D domain. Measure inner profiles, resolution of
   steep features, gauge speeds, mass conservation and constraint convergence.
   A spherical pass does not replace the full 3D acceptance matrix.
5. Use a full puncture treatment or a separately derived inner cut. For excision,
   verify every actual characteristic including gauge/constraint branches on
   every face; a scalar outflow test does not certify Z4c excision.
6. Apply small angular gauge perturbations after the unperturbed cases pass.
   Independent physical gravitational ringdown is optional and requires admitted
   radiative data and extraction; it is not a prerequisite borrowed from the
   deferred milestone 5. A gauge pulse alone is not a physical QNM excitation.

## Gates to finalize from Goal 2

- Three-resolution fields, first derivatives, constraints and inner-profile
  convergence; documented global and puncture-neighborhood norms.
- Predeclared mass/horizon and stationary-drift budgets, transition duration and
  late-time growth limits. Exact-scri endpoint and boundary sensitivity gates.
- Source-aware timestep refinement and actual characteristic checks, especially
  near the trumpet and any inner cut.
- Fixed reference/map/Omega hashes across restarts; no black-hole source
  subtraction, Omega floor, hidden inner exclusions or reset-based stabilization.
- Uniform local CPU acceptance before nontrivial Cartesian AMR and short GPU
  comparisons; keep the shell's actual refinement capabilities explicit.

## Required final task: refine Goal 4

Update [Goal 4](04-kerr-spin05.md) with a justified spinning initial-data/slicing
choice, gauge compatibility, mass/spin diagnostics, inner-boundary policy,
angular/radial resolution needs and exact-scri asymptotics. Decide explicitly
whether the available construction supports stationary preservation, gauge
relaxation, or both. If radiative data remain unavailable, retain that limitation
rather than promising gravitational QNM validation. Update the index/results and
recommend readiness; do not launch the successor automatically.

## Results

No new runs under this plan. Historical spherical successes and 3D failures keep
their original scope and acceptance status.
