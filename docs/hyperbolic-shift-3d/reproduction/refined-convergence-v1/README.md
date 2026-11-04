# RF2 enlarged-center investigation (work in progress)

Branch `codex/refined-convergence`, based on `hyperboloidal_refined`21676d39e.
The persistent goal in GOAL.md is not complete. No RF2 long Aurora evolution
has been submitted. Keep the separate frozen BN1 campaign unchanged.

## What is established

The original sampled boosted initial slice is spacelike and its spatial map
is nonsingular. The minimum normalized spacelike margin Q/Q0 is .9940224;
it is exactly1 in the suspect compact region .3–.4. This does not exclude
an evolved slicing pathology. Prior BN1 endpoint evidence instead identifies
large derivative disagreement between independently evolved copies near the
first shell interface .305. That is a numerical suspect, not causal proof.

`run_initial_audit.py` reproduces independent native initial-geometry checks
at968points per case for stationary and v=.1 configurations. The generalized
native initializer preserves the original4–12M fields exactly on these samples.
For the most enlarged20–40M slice the sampled minimum Q/Q0 is .9684319 and
boost-map detJ is .6943910; all sampled radial/angular map derivatives are
positive. Small absolute Q near scri is normalized by the unboosted Q0 rather
than interpreted as failure. The physical radial map is distinct from Omega;
changing the conformal frame does not replace that map with r/Omega.

The configuration/frame oracle calls the actual Cartesian Z4c RHS. The largest
sampled RHS difference is2.46844e-12; physical constraint frame-invariance and
map inversion checks pass. These are local consistency checks, not a coupled
stability or convergence proof. Production initial data and evolution use the
same configured Omega, native slicing interval, and map normalization k.

## Controlled candidates

| candidate | core half-width | shell overlap radius | flat Omega / end | gauge interval | native areal slice |
|---|---:|---:|---:|---:|---:|
| moderate | .24 | .21 | .24 / .30 | .36–.65 |6M–18M|
| large | .40 | .35 | .40 / .54 | .62–.78 |20M–40M|

Original plateau .5 controls are retained. Matched-plateau controls use
Omega_flat=(1-r_flat²)/2: .4712 and .42. They reduce peak sampled |Omega''|
from140.36 to40.01 (moderate), and79.73 to29.50 (large); the original RF1
narrow join gives171.25. `plot_initial_controls.py` produces the inspected
figure and hashes its raw inputs. The plot shows initial profiles only.

The larger configuration's inscribed Cartesian sphere is about11.86M in
isotropic radius, and gauge onset about18.4M isotropic /19.4M areal. The
20–40M label specifies the native areal slicing interval, not the Cartesian
box size. Its initial-map scale k=.1011686354 differs from original.3061422834.
At N16, unrefined physical Cartesian spacing is .3707M; three refinement levels
reduce it to .0463M. The moderate case with one level gives .0501M, comparable
to original BN1 .0551M. Enlarging the domain without accounting for this map
normalization would substantially underresolve the puncture.

Restoration coefficient4 and .6–.9 collar, KO8, original scri derivative,
fixed effective scri penalty, auxiliary driver and floors remain. Several
parameters change together in these exploratory configurations; isolate them
with controlled comparisons before attribution. The origin-centered .03
positive-chi floor patch has a changed physical size when k changes and needs
explicit audit. Gauge blending in the large case overlaps the restoring collar.

## Native SMR qualification

Reuse AthenaK Cartesian tree/transfer infrastructure with configurable core
bounds, overlap and initial refinement. RF2 identities bind all new parameters;
AMR identities additionally bind `dynamic|fixed|uniform|smr`. Policy changes
cannot silently restart a different formulation. SMR is constructed before
analytic fields are installed. Initial construction bypasses only the regrid
age interval; evolution retains its normal age rules.

The moderate matched-plateau stationary case at N12, dt=.00025, one level
produced120Cartesian leaves and passed two steps. Two MPI ranks through step1,
then one MPI rank through step2, reproduce all eight saved native state,
compensation and event arrays bit-for-bit versus uninterrupted MPI2.
`check_smr_restart.py` records hashes, configuration, topology and finite-array
checks. A changed policy is rejected. Horizon seed differs by2.76e-8 after
finder reinitialization; the checkpoint files therefore are not byte-identical,
even though evolved fields are. Final horizon mass is about.33333339545,
null residuals <=1.23e-15 and no floors occurred. Time is only .0015M.
This does not qualify long evolution, GPU behavior, moving refinement or
three-level SMR. The older `central6-smr-check` predates policy binding and is
retained as development evidence, not a restart input for the final binary.

The earlier uniform `central6-t0` development report incorrectly labeled its
slicing interval4–12M while identity/fields used6–18M. Report metadata is fixed
in source. Do not use that old report as a fully qualified production gate.

## Performance and remaining work

Existing optimized halo construction, sparse exchange, device evolution and
native AMR machinery are reused. Local MPI2 Serial measurements (not Aurora
scaling) show approximately10.3s per step; cumulative bulk halo time9.83s
versus RHS7.73s across the earlier two-step check. Setup is about20s and
performing full diagnostics/horizon finding every step adds seconds per step.
Timers overlap as stated in their native records; do not sum nested and
exclusive timers or extrapolate this CPU fixture to GPU costs.

Before a production SMR run, extend the qualification driver with scheduled
rather than every-step diagnostics, walltime checkpoint disposition, complete
candidate/exterior histories and run provenance. Measure improvements against
a fixed-state result, preserving physical operators and failure checks. The
current driver is a qualification harness, not yet the production campaign.
Next causal controls must test interface derivative/overlap behavior and
monitor evolved slice/metric regularity. Stationary qualification precedes
boosted runs, with resolved puncture spacing and controlled timestep, followed
by common-domain exterior and field-resolution analysis. Survival alone is
not convergence.

## Production driver extension

Configured AMR accepts three optional trailing arguments:
`max_seconds diagnostic_interval target_time`. With these present, the target
is coordinate time (10M=10/3), the requested dt is held fixed after its
characteristic-bound check, and the final step is shortened to hit the target.
Diagnostics are scheduled on an absolute coordinate-time grid, with initial,
walltime and final rows preserved. A healthy walltime checkpoint returns3;
a numerical or I/O failure returns2 and cannot be used as a continuation.
The old per-step qualification invocation remains available. The synthetic
`dynamic` regrid fixture is rejected in production mode; production currently
uses fixed SMR, not a claimed moving adaptive criterion.

One native constraint pass accumulates candidate-enclosing and fixed cuts
r>.1,.3,.6, with H/M/Mr and shift/D derivative maxima. Full candidate acceptance,
expansion and mass gaps are recorded. Scri pole/clock values are cumulative
maxima. Counter snapshots are cumulative across restarts, with bulk, surface
and boundary events separate; segment differences must be taken rather than
summing cumulative reports. The native checkpoint retains detailed event data.
A restart topology reconstruction now explicitly permits initial multi-level
tree construction without the evolution regrid-age delay.

`check_production_controls.py` verifies identical native/compensation/event
arrays against the pre-extension two-step stationary test, both with reduced
diagnostic cadence and after a healthy walltime checkpoint (MPI2 to MPI1).
The horizon seed is not bit-identical because finder calls differ. This is
not a long-time or GPU qualification. Source/build provenance for the next
Aurora experiment is separate from all frozen RF1 and BN1 snapshots.
