# Handoff: shared-equation 1D/3D slicing and cost optimization

## Evidence and scope

The user wants inexpensive local spherical experiments before more expensive
3D runs. The 1D solver MUST solve the spherical restriction of the same
underlying equations as 3D. Preserve a configurable central region matching
the Cartesian Z4c/telegrapher implementation, smoothly joined to a regular
hyperboloidal exterior. Optimize accuracy at fixed radial point count AND
cost at fixed accuracy, including less restrictive stable timesteps.

Reference worktree: /Users/hz0693/.codex/worktrees/refined-convergence/hyperboloidal.
Reference branch: codex/refined-convergence; inspected scientific/evidence base
0e71080c382f53b0aa848f748bb0e54240abc26f. Frozen latest Aurora scientific
source d98e8d9e7f6131a4af55afe96ef7589a73d07b9e remains immutable.
Cartesian reference: origin/project/telegrapher_lapse,
b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a.
Use a NEW worktree and codex/spherical-gauge-optimization branch from the
reference branch; do not modify the reference worktree or its frozen campaigns.
This handoff does not authorize new Aurora allocations or resuming the paused
parent goal. Start locally. Do not create another thread automatically.

Current findings:

- Enlarged reference central region 20M, flat Omega=.42 to compact r=.4,
  Omega join to.54, inner/outer gauge blend .62–.78, restoration4 over.6–.9.
  Native C6 slicing changes over areal 20–40M, mapped to r=.64029–.92781.
- Conformal lapse baralpha=Omega*alpha_phys: its exact stationary minimum
  .1614 has physical lapse1.0624. Low plotted conformal lapse is not equivalent
  to physical lapse collapse. Its sharp gradient near.909 remains real in
  compact coordinates: peak29.48, halfmax width.02137.
- At fixed central scale, an analysis-only compact slicing switch ending.97
  spreads the tilt to approximately 98.48M, reducing that gradient to 13.53.
  Exact initial geometry and independent radial differences improve; native
  boosted production implementation/evolved accuracy are not yet demonstrated.
- Short boosted block-layout runs have over99% of exterior squared constraints
  near r=.94–.97, around the mesh transition. Larger24³ blocks improve per-cell
  throughput 1.386× but this layout has 2.478× more cells and costs 1.834× more.
- The current radial CFL limiter is nearly one-dimensional: replacing the sum
  by a Euclidean norm offers only 1.0095× on the measured state. Dense radial
  spacing, sources and boundary treatment need investigation.
- D is evolved in the bulk via L D=L Lambda-D and drives the outer shift;
  inner shift is direct Gamma-driver. At scri D is dependent on the projected
  shift and actual geometric RHS, not independently prescribable. Nullness
  alone does not prove auxiliary-PDE compatibility or boundary well-posedness.
- Two later short interface controls 9041071 are remotely admitted: grading
  radial spacing lowers H/M about 5.3×/3.9× at.075M; moving the angular seam
  alone does not. Radial grading also changes the scri penalty rate. Full
  local preservation/native spatial follow-up is separate parent work; do
  not treat these as 10M stability or convergence evidence or duplicate runs.

Read GOAL.md, state.json, LARGE-BLOCK-CFL.md, COMPACT-SLICING-STUDY.md,
TAIL-INTERFACE-ATTRIBUTION.md and LAPSE-REGULARITY-OPTIONS.md in this directory.
The requested approximately 4× stable measured GPU-hour improvement before a
boosted 10M three-resolution campaign remains unmet. 1D CPU speedups are not
GPU-hour measurements, nor authorization to bypass that later gate.

## Copyable task prompt

Implement and test a shared-equation spherical optimization laboratory in a
new worktree, starting from codex/refined-convergence. The aim is a regular,
accurate and inexpensive hyperboloidal puncture formulation that transfers to
3D, with an enlarged central Z4c region suitable eventually for a binary.
Do not optimize a different set of equations just because it is easier in 1D.

1. **Establish equation identity before parameter sweeps.** Reuse
   tst/hyperbolic_shift_1d, tst/elliptic_shift_1d, spherical_tensor.hpp,
   ConformalRHS, HyperbolicGauge, HyperbolicAuxiliaryRate, ConformalProfile,
   NativeWormhole and the production scri geometry/gauge closure where applicable.
   The existing Python/C++ 1D path is NOT already matched: spherical_api.cpp
   hard-codes Omega=(1-r²)/2 and older reference/gauge/damping choices, and
   its variable/floor/derivative policies differ. Preserve old reproductions.
   Add an explicitly identified current-formulation mode whose geometry,
   gauge coefficients, sources, conformal frame, and limiting equations come
   from shared C++ routines. Keep Python for orchestration/plots as useful.
   Avoid maintaining a second independently edited Python RHS. Distinguish
   continuum equivalence from differences in spatial discretization.

2. **Use one central-region configuration.** Introduce a clear user parameter
   such as inner_z4c_radius_M, defined as reference areal radius in the spherical
   initial map (default 20M). Derive its compact coordinate, Omega plateau end,
   gauge-blend onset/end, initial slicing transition and restoring/reference
   source windows from one configuration object, with documented relative
   width ratios rather than scattered unrelated hard-coded radii. Preserve
   independent advanced choices only when explicit and validated. Omega is
   constant and the inner gauge exactly matches the selected Cartesian Z4c
   parameters after the required coordinate/conformal rescalings throughout
   the requested inner sphere; all joins start outside it and are smooth.
   For 3D expose the same configuration, report whether the sphere fits inside
   the Cartesian box, and coordinate shell placement with these joins.
   A reference areal radius is not an invariant binary orbital radius; report
   its coordinate/physical interpretation and leave room for orbital/gauge
   motion. Reject impossible ordering, nonmonotone maps or singular profiles.
   Derive transformations of lapse, chi, curvature, Gamma coefficients and
   damping units; a constant Omega does not excuse inconsistent initial data.

3. **Prove the reduction is faithful.** Use nontrivial spherical jets with
   evolving radial/tangential metric, shift, lapse, independent Lambda, Theta
   and D. Compare every reduced RHS component with the shared 3D tensor kernel,
   including off-constraint data, Omega derivatives, all sources and damping.
   Do not replace evolved Lambda by the metric-built connection or reset
   constraints. Check flat-core equivalence against the actual telegrapher
   branch, not a remembered gauge equation. Verify initial geometry/map,
   conformal-frame transforms, independent continuum constraints, and scri
   limits. Include changing-geometry/nonzero-gradient boundary data and the
   differential trace compatibility of D-Lambda. Keep geometric constraints
   distinct from null projection and algebraic closure checks.

4. **Optimize a small, attributable candidate family.** First reproduce the
   current spherical formulation. Separately vary: (a) slicing/map width and
   shape at fixed central physical scale; (b) consistent lapse/shift reference
   sources and normalized/background perturbation variables; (c) gauge-speed
   and damping profiles; (d) radial grading/interface treatment; (e) scri
   penalties. Do not change all at once. Carry nonlinear evolution of all
   geometry fields. Candidate changes must be expressible in the shared 3D
   formulation, not depend on an elliptic spherical shortcut. Preserve a
   correct initial-data embedding, all profile derivatives and compatible
   scri closure. Any well-balanced correction must include the actual
   continuum background RHS; do not artificially freeze a nonstationary
   puncture or future boosted solution. Distinguish a variable change from
   an equation change. Don't simply smooth D, remove necessary lapse/chi
   couplings, or rely on increased KO/floors to conceal unstable behavior.

5. **Measure characteristics and the timestep restriction.** Audit the full
   reduced principal symbol, eigenvector completeness/conditioning and speed
   coincidences across the domain and during evolution. A singular variable
   change at puncture/scri is not a regularity proof. Check full 3D transverse
   sectors analytically or with pointwise symbols before recommending a
   candidate; spherical evolution alone cannot exercise them. Compute mapped
   coordinate speeds, shift advection, telegrapher relaxation, KO and penalty
   restrictions. Use the semidiscrete operator/RK4 stability polynomial and
   nonnormal amplification checks on manageable linearizations, then nonlinear
   timestep sweeps. The existing 1D .4 guard is a policy, not a proven optimum;
   study separate opt-in limits without silently deleting guards.

   Test effective characteristic CFL targets around .1,.2,.3,.4,.5 and above
   only when justified. Report C_char=dt*max(|lambda_coord|/local_dr), dt/M,
   min_dr/M, steps/M and the active limiter, plus any nominal dt/min_dr label.
   Reference z4c_newdt.cpp uses min directional dx on its ordinary path, but
   telegrapher mode uses max(1,abs(lapse_advect*beta_i)+sqrt(kappa/tau*gamma^ii))
   and a tau bound. Do NOT infer every speed equals one. Even physical radial
   coordinate speeds are -beta^r +/- baralpha*sqrt(chi*gtilde^rr); mapped gauge
   modes and stiff sources add restrictions. Changing time normalization is
   not a speedup. Compare initial and worst-evolved speeds. If boundary
   stiffness dominates, test consistently derived penalty-rate alternatives
   with separate identities and perturbation/boundary-reflection checks.

6. **Evaluate accuracy at fixed points and cost at fixed error.** Use staged
   small local runs, then at least three radial resolutions spanning a useful
   range (e.g. 64/128/256, with 512 only when justified). Preserve the same
   central physical scale and meaningful exterior diagnostic domain; compare
   at common physical/reference coordinates with explicit interpolation errors.
   Use exact stationary data only when it is stationary under the complete
   selected gauge, regular gauge/constraint perturbations, and the actual
   wormhole-to-puncture gauge transient. Continue finalists through at least 10M;
   a cheap longer check can test delayed growth. Preserve origin parity or
   puncture-chart treatment and verify that it does not hide exterior errors.
   Don't count divergent puncture-interior norms as exterior failure.

   Record fixed-exterior H/M/Theta/Z norms, regional contributions and maxima,
   full and physical lapse, chi, metric, shift, D, their unsmoothed derivatives,
   feature widths in cells, outgoing/ingoing gauge disturbances, scri null,
   regularity/pole/clock and auxiliary-compatibility residuals, horizon expansion
   and accepted mass with gaps, floor/sign counts, actual steps, wall time and
   RHS cost. Separate evolving gauge transients from growing numerical errors.
   Run temporal refinement for finalists so dt error cannot masquerade as
   spatial convergence. Use constraints and independent field differences;
   inspect local errors near joins as well as global fitted orders. Don't
   declare asymptotic convergence from survival or one favorable pair.

   Rank a Pareto frontier: exterior accuracy at equal radial point count;
   wall time/RHS evaluations to a specified error at the same 10M duration;
   and maximum robust stable dt with an accuracy margin. Count production
   diagnostics consistently and separate setup from stepping. A faster
   variable with larger systematic error is not an unqualified improvement.

7. **Preserve evidence and define the transfer gate.** Freeze configs/source
   identities and reproduce plots/tables from saved data. Failures are data;
   do not silently restart with changed parameters or omit failed candidates.
   Commit/push the separate branch. Deliver the equation/config mapping,
   spherical-vs-tensor parity results, characteristic/stability audit,
   time/resolution sweeps, accuracy-cost curves, selected candidate and
   unresolved issues. No new Aurora runs without explicit authorization.
   Before 3D adoption, require shared-kernel parity, a small spherical 3D
   comparison with the 1D solution, then nonspherical/angular perturbations,
   interface/MPI checks, and the actual v=.1 boosted test. A boosted puncture
   breaks spherical symmetry and cannot be validated by radial 1D evolution.
   The final boosted 10M three-resolution convergence and measured GPU-hour
   performance gates remain separate requirements, not claims of this study.

The desired result is a physically consistent, smooth, inexpensive formulation
with demonstrated radial accuracy and a clear 3D transfer path—not merely a
larger advertised CFL or an artificially stationary spherical solution.
