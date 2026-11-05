# Handoff requirement audit and 3D transfer gate

This audit maps the seven requirements in
[SPHERICAL-OPTIMIZATION-HANDOFF.md](../hyperbolic-shift-3d/reproduction/refined-convergence-v1/SPHERICAL-OPTIMIZATION-HANDOFF.md)
to the local implementation and retained evidence. The local study selects a
finite-duration spherical working policy; it does not certify a globally
optimal gauge, all-time well-posedness or production 3D stability.

| Requirement | Implementation and evidence | Scope / remaining transfer issue |
|---|---|---|
| 1. Equation identity before sweeps | `current_spherical.hpp` lifts radial total fields and jets into shared `ConformalRHS`, damping, gauge, source and auxiliary kernels. Initial tensor/Cartesian parity preceded parameter sweeps. Historical 1D modes remain intact. [README](README.md) records the equation mapping. | Spatial stencils, algebraic chart and boundary numerical policy are explicit local choices, not already identical to 3D. |
| 2. One central configuration | `CentralZ4cConfiguration::inner_z4c_radius_M` defaults to 20M. All conformal, gauge, slicing and source windows derive from it and validated width ratios. The 3D parser exposes the same key and rejects a sphere that does not fit. | Reference areal radius is an initial-map label, not binary separation or a gauge-invariant evolved radius. Existing production box/shell layout needs redesign for this sphere. |
| 3. Faithful reduction | Nontrivial off-constraint jets compare every reduced component with the tensor kernel; initial scaled error 6.83e−13. Actual Cartesian `CalcRHS` reference errors 2.47e−12/2.19e−13 above/below the correctly transformed guard. Exact ten-variable algebraic chart has tangential RHS agreement 5.68e−14 and does not reset H/M/Theta/Z/Lambda. Moving-geometry scri identities and independent differential-D probes are separate. | Finite-precision fixtures establish numerical equivalence on tested inputs, not a formal proof over all states. Dependent-D algebraic closure is not differential compatibility. |
| 4. Attributable candidates | [COMPACT-PROFILE](COMPACT-PROFILE.md), [REFERENCE-VARIABLES](REFERENCE-VARIABLES.md), README grading controls, [BOUNDARY-TRIALS](BOUNDARY-TRIALS.md), [COMPATIBLE-LAPSE](COMPATIBLE-LAPSE.md), [OUTER-LAPSE](OUTER-LAPSE.md) and [PENALTY-TRADEOFF](PENALTY-TRADEOFF.md) separate map, representation, stencil, damping and penalty changes. | Earlier graded families are rejected/diagnostic controls; the selected final policy is uniform. No claim that every possible grading or gauge was optimized. |
| 5. Characteristics and RK4 | Free twelve-field Jordan defect, exact-chart invariance, root-located speed coincidences, separated-speed profiles and full 26-field radial/transverse/oblique symbols are retained. [RK4-LIMITS](RK4-LIMITS.md) and penalty report include saved Jacobians, RK4 polynomial, nonnormal RK/exact comparison, stiff-mode sensitivities and nonlinear failures. | Initial max coordinate speed 2.414213562373095. Working initial CFL .68 has about 12.45% sampled frozen margin; successful .75 is not promoted without the same margin. Point symbols are not nonspherical evolution or uniform endpoint hyperbolicity. |
| 6. Accuracy and cost | Same 20M central scale, common reference R/M25–100 with interpolation sensitivity, nonlinear 10M transient, early fourth-order time refinement, field/constraint spatial differences through N1024 for the continuum finalist and matched-penalty controls through N512. N128/256 pulse tests extend to40M. [COST-PROTOCOL](COST-PROTOCOL.md) fixes serial repeated cost and separate boundary screens; [FINAL-COST](FINAL-COST.md) reports 27 successful measurements with bitwise repeated states and 4.405–4.411× boundary-screen cost ratios. | Constraint E is an indicator, not certified solution error. Coarse errors and the late gauge-response upturn are retained. CPU ratios are local measurements, not GPU-hour gains. |
| 7. Evidence and transfer | Immutable run manifests, source archives/hashes, binary identities, last accepted and failed states, saved arrays and plot scripts. Branch is isolated from `codex/refined-convergence` and contains `718ad1fdd`; no parent campaign or Aurora work. | The transfer gate below remains separate from local completion. |

## Selected local configuration

Central20M; C6 compact slicing endpoint .97 with matched central anchor;
separated gauge speeds; exact ten-variable determinant/tracefree chart;
chi reference power two; reference jets at step .000125 with exact scri tail;
analytic shared normal; compatible-lapse fraction one in the gauge join;
outgoing C=D−Lambda width-five stencil with shift-only boundary normal;
outer lapse damping .25; penalty .5; KO .5; uniform grid; no state floors.
Use explicit experimental fixed dt for initial characteristic CFL .68.
The conservative guards and default coefficients remain available unchanged.

The lapse-damping change is a shared continuum/source change. Reference-r2 and
exact algebraic coordinates preserve the represented equations. Composed
lapse D1D1 is a consistent spatial-policy change, localized to the gauge join;
it must be implemented and compared explicitly in 3D. The outgoing shift-normal
closure also requires a 3D numerical implementation. No spherical elliptic
solve, artificial stationary-background subtraction, constraint reset or
increased KO conceals the rejected failures.

## Preserved failures and unresolved scientific limits

- Constant-driver speed coincidences and the free twelve-field extension have
  incomplete eigenspaces. Separated speeds alone do not repair the free extension.
- Wider areal maps and graded controls can improve one exterior norm while
  damaging boundary defects or horizon diagnostics. Matched-rate two-end
  grading fails at nearly the same time after reducing dt by four.
- The r4 representation, coupled/neutral lapse-normal lifts and certain
  outgoing auxiliary closures are rejected. Stronger auxiliary damping does
  not cure the discrete interior gauge mode.
- Direct lapse D2 in the blend creates a grid-growing short-wave branch; the
  compatible-lapse spatial policy removes that measured branch.
- Outer lapse damping one admits a fractional near-scri mode with metric/lapse
  power about 1.772. Damping .25 moves the predicted weighted power to 3, with
  observed boundary derivative convergence. This does not prove all orders of
  smoothness or nonlinear well-posedness.
- RK4 failures are retained on both sides of the measured boundary stiffness
  transition. Initial/evolved frozen spectra contain positive-growth modes
  and nonnormal amplification; no contractivity claim is made.
- The N256 gauge pulse response turns upward late between 32M and 40M. Timestep
  and amplitude controls resolve it. Its small finite-duration magnitude is
  sufficient for the reported finite-duration comparison, not all-time decay.
- Near-puncture errors, algebraic tangent residuals, interpolation errors,
  horizon acceptance gaps and boundary residuals remain separate diagnostics.
  The exterior domain excludes the puncture; no divergent interior norm is
  silently treated as an exterior failure or suppressed by floors.

## Required before any 3D adoption

1. Port the exact algebraic/reference, compatible-lapse and outgoing-normal
   numerical policies into the shared 3D path, including interfaces and device
   kernels; retain default parity and repeat nontrivial source/boundary parity.
2. Design a Cartesian box/shell layout containing the full central 20M sphere
   with resolved joins. Compare a small spherical 3D evolution against these
   radial solutions, including fields, physical constraints and scri defects.
3. Exercise angular/nonspherical perturbations, mesh interfaces, MPI exchanges,
   and device execution; pointwise transverse symbols do not replace these.
4. Run the actual v=.1 boosted case under separate authorization. Compact
   spherical slicing is currently explicitly unsupported for boosted data.
5. Meet the separate boosted 10M three-resolution convergence and measured
   GPU-hour performance gates before the parent campaign adopts any policy.

No Aurora allocations or parent-campaign changes are authorized by this report.
