# Local spherical optimization laboratory

Branch: `codex/spherical-gauge-optimization`, based on `718ad1fdd` of
`codex/refined-convergence`. This directory is a separate local experiment.
The parent campaign, frozen Aurora snapshots and remote jobs are untouched.
**Work in progress: no candidate is admitted, no stability/cost improvement is
established, and no spherical result establishes boosted or 3D stability.**

Current results are in [SHIFT-NORMAL-BOUNDARY.md](SHIFT-NORMAL-BOUNDARY.md):
the outgoing auxiliary/shift-normal policy improves the boundary defect and
preserves exterior convergence, but leaves the late gauge response unchanged.
[CONSTRAINT-PROPAGATION.md](CONSTRAINT-PROPAGATION.md) checks the shared interior
equations on smooth stationary and perturbed-gauge data. The next stability
investigation targets a resolution-dependent mode in the interior gauge blend.

Latest auxiliary-damping and rejected outgoing-boundary experiments are in
[BOUNDARY-TRIALS.md](BOUNDARY-TRIALS.md). Stronger damping does not cure the
late response; the outgoing trace closure has a grid-scaled unstable mode.
The controlled serial measurements in [ACCURACY-COST.md](ACCURACY-COST.md)
give an exterior-only frontier. No finite fixed-target speedup ratio is
established because the areal control misses all tested accuracy targets.

## Equation and variable mapping

`src/z4c/hyperboloidal/current_spherical.hpp` lifts **total** radial fields to
Cartesian tensor jets at `(r,0,0)`. Its twelve slots are
`chi, gr, gt, Ar, At, Khat, Theta, Lambda, alpha_bar, beta, B, D`.
Radial and tangential metric/curvature components evolve independently. The
adapter neither substitutes the metric-built connection for Lambda nor resets
Theta, Z, determinant or tracefree constraints. All bulk evolution comes from
`ConformalRHS`, `HyperbolicDamping` (including `CartesianInnerAdjustment`),
`HyperbolicGauge`, `ApplyStationaryGaugeSource` and `HyperbolicAuxiliaryRate`.
The old Python/C++ reproductions remain unchanged. Python constructs spatial
stencils, selects numerical dissipation, integrates RK4 and records diagnostics;
it does not implement another continuum geometric or gauge RHS.

The reference Cartesian source is commit
`b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a`. The equations in its
`z4c_calcrhs.cpp` are unchanged in this worktree's ordinary Cartesian path;
the differences add a hyperboloidal dispatch and a mapped-stencil adapter.
The local reference check runs the actual `Z4c::CalcRHS`, with ordinary
Cartesian stencils, polynomial nonconstant metric/lapse/chi and nonzero
Theta/Lambda/B. It does not compare against a remembered telegrapher formula.

With the same coordinate time, `x=k X`, and constant core Omega `O`, the
coordinate/conformal transformations are

- `alpha_bar = O alpha_X`, `chi_bar = k^2 chi_X/O^2`;
- `gtilde_x = gtilde_X`, `Abar_x = Atilde_X/O`;
- `Khat_x = Khat_X`, `Theta_x = Theta_X`;
- `beta_x = k beta_X`, `Lambda_x = Lambda_X/k`, `B_x = B_X/k`;
- the inner Gamma coefficient is `k^2` times the Cartesian coefficient;
- the inner chi coefficient guard is `k^2 chi_div_floor_X`;
- tau, kappa, eta and kappa1 retain their time units under this spatial change.

For the reference mass `M=1/3`, the mass-scaled deck has tau=`.1M`,
kappa=`.2M`, eta=`2/M`, kappa1=`.01/M`, kappa2=0. The inner lapse uses the
original partial-derivative telegrapher coupling, not a covariant divergence.
The actual Cartesian chi guard substitutes coefficients while retaining raw
field derivatives; it is not the differentiated function `max(chi, floor)`.
The new opt-in configuration reproduces that policy in the core and smoothly
turns it off with the inner weight. This is explicitly a numerical
regularization. Physical diagnostics always use the unguarded evolved state.
Legacy configured targets retain their previous arithmetic and guard policy.

## One central region

`CentralZ4cConfiguration::inner_z4c_radius_M` defaults to 20. It denotes the
**outer-branch reference areal radius of the initial spherical map**, not proper
distance, a binary orbital radius, or an invariant boundary during evolution.
The C6 areal slicing interval defaults to `[R0,2 R0]`. `NativeWormhole` determines
its compact onset `r0` and isotropic scale `k`; all conformal and gauge joins
start at or outside `r0`. For `L=1-r0` the default windows are:

| Quantity | Interval |
|---|---|
| Omega plateau | `0 <= r <= r0`, Omega=.42 |
| Omega C6 join | `r0` to `r0+.25L` |
| Inner/outer gauge and coefficient window | `r0+.35L` to `r0+.65L` |
| Initial lapse reference blend | `r0` to `r0+.35L` |
| Shift restoration | `r0+.35L` to `r0+.85L` |
| Prescribed outer reference sources | weighted by the same outer gauge weight |

Advanced width ratios are explicit C++ configuration fields and undergo finite,
ordering, positivity and native-map checks. A different slicing endpoint also
changes the isotropic scale, which is reported and propagated into Gamma/guard
coefficients; the physical central scale stays fixed. The historical baseline
is separately labelled `mode=0`, preserving flat .4–.54, gauge .62–.78 and
restoration .6–.9. It does **not** make the whole reference 20M sphere Cartesian.

The 3D refined parser accepts the same `inner_z4c_radius_M` key and rejects
conflicting independent transition settings. It verifies the sphere fits the
specified Cartesian box and retains the shell-resolution/buffer checks.
At default 20M, `r0=0.64029066119359435`; the historical half-width .4 box does
not contain it. The existing cube/shell topology also limits supported box size;
this is a real 3D-transfer issue, not grounds to silently reduce the sphere.
No new 3D layout has been admitted. Shell placement must satisfy the existing
resolved-window checks; it is not automatically qualified by the new key.

## Boundary and discretization scope

The scri adapter follows the production geometric pole derivative, regularity
penalties, fixed clock, null shift projection and moving-geometry dependent D
closure. The radial normal derivative uses a seven-point polynomial numerator
stencil; this is a spatial policy different from 3D interpolation. A changing
metric/nonzero-gradient fixture verifies the null time derivative and shift
algebraic closure. **It does not prove differential D compatibility.** The
independent diagnostic differentiates the dependent-D functional along the
semidiscrete flow and compares it to the actual auxiliary PDE trace.

The initial audit already exposes underresolution: at N=64 the normal operator
and its differential compatibility probe sample the steep slicing transition.
Its initial D compatibility defect is about 3.06e4. At N=128 it is about 1.45;
at N=256 about 4.84e-8 in the original pilot. These are diagnostics, not
boundary-convergence or well-posedness claims; temporal probe refinement and
nonstationary perturbation tests remain necessary.

Bulk derivatives use seven-point polynomial stencils, with sign-selected biased
component advection. Puncture data use one-sided stencils at the excluded
puncture, not regular-center reflection. KO8 is an explicit fixed numerical
policy tapered inside r=.8. On graded grids its minimum-spacing normalization
is disclosed and must be included in operator stability analysis. No
background-RHS subtraction, elliptic shortcut or automatic restart is used.
Optional state floors are explicit and counted, with maximum activation radius;
they are off in initial pilots. A failure retains the last accepted state and,
in the newer runner, the attempted stage.

## Reproduce locally

```
cmake -S tst/spherical_optimization -B build-spherical -DCMAKE_BUILD_TYPE=Release \
  -DKOKKOS_SOURCE=/path/to/local/kokkos
cmake --build build-spherical -j 4
ctest --test-dir build-spherical --output-on-failure
python3 -m venv build-spherical/python
build-spherical/python/bin/pip install numpy scipy matplotlib mpmath==1.3.0
build-spherical/python/bin/python tst/spherical_optimization/audit.py \
  --library build-spherical/libcurrent_spherical.dylib --output fresh-audit
build-spherical/python/bin/python tst/spherical_optimization/run.py \
  --library build-spherical/libcurrent_spherical.dylib --mode 1 --n 128 \
  --duration .1 --output fresh-run
```

On Linux use the generated `.so`. Kokkos source can be the repository submodule;
source outside this worktree is read-only and build artifacts stay local. The
runner refuses an existing destination. Each experiment records configuration,
commit, source hashes, a source archive, binary hash, states, unsmoothed field
jets, regional constraints, timing, step counts and actual speed estimates.
The first two pilot archives were backfilled and verified against their original
hash manifests before any source changes. Qualification v1's binary is preserved.

The ordinary timestep retains an explicit .4 characteristic-CFL policy guard,
a .5 scri penalty bound and a telegrapher relaxation bound. Larger characteristic
CFL or fixed-dt tests require `--experimental-timestep`; no legacy guard is
removed. The runner reports `C_char=dt*max(speed/dr)`, dt/M, min dr/M, steps/M,
nominal dt/min dr, initial/worst evaluated speeds and active limiter. Completed
runs are labelled **completed_unqualified**, never automatically stable.

The full 26-variable 3D point symbol includes transverse sectors and the
algebraic-constraint tangent space. It is audited in the original conformal
variables at interior points. Endpoints and near-puncture conditioning require
separate interpretation. Raw eigenvectors returned for repeated eigenvalues can
be arbitrarily ill-conditioned; the second audit uses SVD bases for each
repeated eigenspace and explicitly checks algebraic/geometric multiplicity.
Speed coincidences are root-located rather than inferred from a sparse grid.

## Remaining admission gates

See STATE.json for the current next steps. Required work includes a qualified
nonstationary scri/differential-auxiliary check; perturbation and wormhole
transient evolution; attributable slicing/reference-variable/speed/grading/
penalty controls; temporal and three-resolution field/constraint convergence;
full-symbol conditioning during evolution; RK4 semidiscrete and nonnormal
analysis followed by nonlinear timestep sweeps; 10M finalists; reproducible
plots and a fixed-accuracy cost frontier including all failures.

Before 3D adoption: shared-kernel parity, a small spherical 3D/1D comparison,
angular perturbations, interfaces/MPI, then the actual v=.1 boosted test.
Spherical CPU timings do not satisfy the separate fourfold measured GPU-hour
or boosted 10M three-resolution campaign gates.

## First local checkpoint

The off-constraint tensor reduction maximum scaled error is 6.83e-13. The
actual Cartesian reference errors are 2.47e-12 above the guard and 2.19e-13
below the correctly rescaled guard. The new 3D key explicitly rejects a 20M
sphere in the default half-width .18 box.

The constant-driver initial symbol has root-located lapse/shift coincidences at
r≈.066914 and .819594 in the central20 configuration (the historical second
root is ≈.696689). At each root the radial and transverse symbols have
multiplicity two but numerical nullity one. The separated-speed control has
no such roots in this audit. It changes the inner gauge and is not the original
g1 Cartesian reference. A complete uniform hyperbolicity proof is not claimed.
At defective roots the reported finite SVD-basis condition is **not** a
condition number of a diagonalizer: the nonzero eigenspace residual and nullity
failure identify that no complete eigenbasis was obtained.

At t=1M, on common reference areal R/M=25–100 with uniform log-R weighting,
the maximum dimensionless H/M/Theta/Z RMS indicator is approximately .1095,
.05972 and .03900 for 64/128/256 separated-speed points. The one-triplet field
orders vary about 1.04–2.51, with material interpolation disagreement; this does
not establish asymptotic convergence. At 128 points, moving the native slicing
endpoint from 40M to 80M lowers this indicator to .02784 at similar 1M CPU cost,
but the scri stencil becomes underresolved. Both effects are retained in
`evidence/comparison-1M-v2/accuracy-cost.png` and `transients.png`.

The wider 128-point control completed 10M in 147.97 CPU seconds and 6641 steps,
without state floors, at maximum stage characteristic CFL .154666. It is
**rejected for accuracy**: final clock residual 4.552, pole residual 7.914,
determinant residual .1667, tracefree residual .8891 and inferred horizon mass
.28708 versus 1/3. Horizon interpolation at this resolution is itself limited;
its accepted sign crossing is not a guarantee of accuracy. No candidate or
robust timestep is admitted. All saved states and source identities remain.

The v1/v2 runner recorded endpoint-based characteristic CFL after each step;
its separately recorded worst evaluated speed supplies a conservative bound
on these uniform-grid pilots. The v3 10M run records the maximum over all RK
stages and the worst speed/dr. Earlier artifacts are not silently relabelled.

An important unresolved issue is algebraic constraint handling: the current
free twelve-field evolution accumulates determinant/tracefree errors, whereas
the 26-variable full-3D point-symbol audit is restricted to their tangent
space. The next gate must examine the full free reduced symbol and/or an
explicit algebraic variable chart with proven tangential equations. Such a
chart must never replace evolved Lambda or reset H, M, Theta or Z. The
production spatial/algebraic policies must be distinguished from continuum
kernel equivalence before any optimization claim.

## Algebraic chart and reference-derivative checkpoint

The complete **free twelve-field radial symbol** now includes the two algebraic
sectors omitted from the previous tangent-space audit. At six sampled radii
(.02, .2, .7, .85, .95, .99), its advective eigenvalue has multiplicity four
but nullity three. Thus the free extension is not a suitable evolution
baseline for a well-conditioned formulation, even with separated gauge speeds.
The old free-field runs remain preserved as rejected diagnostic extensions.

The opt-in `--algebraic-chart` evolves ten independent coordinates, with
`gt=gr^(-1/2)` and `At=-Ar*gt/(2gr)`. Tangential values and their first/second
jets are obtained by exact C++ chain rules. Tangential fields are not separately
advanced and subsequently repaired. Lambda, Theta, H, M and Z are never reset
or inferred from constraints. Arbitrary nontrivial off-constraint jets give
continuum tangential-RHS agreement to 5.68e-14. The ten-dimensional principal
subspace is invariant to 8.33e-17 at the sampled points, and its separated-speed
symbols have complete numerical eigenspaces there. This is an interior point
check, not a uniform endpoint hyperbolicity proof. At 1M the algebraic
residuals remain at roundoff, while differential constraints are nonzero.

The `--normal-policy analytic` option uses the existing production
`TensorPoleNormalJet` AD kernel instead of radial numerator differences. This
improves the boundary tangency discrepancy but barely changes the 128-point
exterior constraint error by itself. The discrepancy of the omitted tangential
boundary rate is recorded explicitly; algebraic identities alone do not certify
boundary compatibility.

The additional `--reference-jets` option differences deviations from the initial
native fields and adds high-order reference derivatives. It changes the spatial
representation, not the continuum RHS: the full shared kernels still evaluate
all sources. The actual initial core RHS reaches 33.83, so the nonstationary
puncture/gauge transient has not been frozen. Reference derivatives are currently
**numerical**, prepared with seven-point stencils at h=.001 and .0005; their
saved difference is a setup-error diagnostic. The initial outer RHS maximum
is 3.40e-4, not declared zero. Refining this preparation or replacing it with
analytic shared initial jets remains necessary before tight-error admission.

`physical_constraints.py` independently derives spherical H/M/Theta/Z from
`dl²=A dr²+B² dOmega²`, including the full physical curvature trace. It agrees
with the shared kernel on the initial chart to 1.47e-14 (H) and 1.88e-15
(physical momentum). Reports also retain a separate plain-nodal differentiation
estimate, so the reference representation's derivatives are not the only
available diagnostic. For free-field data with nonzero trace-A, the old kernel
constraint formulas assume algebraic identities that no longer hold; their
nominal norms must not be treated as independent physical accuracy evidence.

The common R_reference/M=25–100, t=1M comparison gives:

| Representation | Points | CPU seconds | Max dimensionless physical H/M/Theta/Z RMS |
|---|---:|---:|---:|
| Chart, sampled normal |128|15.73|.07149|
| Chart, analytic normal |128|15.84|.07149|
| Chart, reference derivatives, analytic normal |64|4.20|.007007|
| Same |128|15.81|.003959|
| Same |256|61.79|.001990|

These are preliminary one-duration CPU observations, not a fixed-error 10M
frontier. Interpolation disagreements are saved alongside the norms in
`evidence/chart-comparison-v6/comparison.json`; the corresponding plot is
`chart-comparison.png`. The differential D trace defect at 1M is approximately
-138.16, -4.168 and +2.347 at 64/128/256. Three directional time-probe sizes give
consistent values, so this is not explained by the probe epsilon. No candidate
is admitted. The next boundary investigation must retain the actual auxiliary
PDE and solve compatible normal-derivative conditions, rather than equating
nullness or algebraic D reconstruction with differential compatibility.

An initial-data hypothesis was explicitly falsified during this checkpoint:
`NativeWormhole::PhysicalLapse` already smoothly blends the precollapsed and
outer lapse. The suspected finite jump at the central slicing onset does not
exist. The gap shrinks from 6.391e-6 at h=1e-4 to 6.391e-9 at h=1e-7, and the
one-sided derivative gap approaches roundoff. An unnecessary alternate blend
was reverted completely. Runs labelled v5 preserve that unadopted blend in
their source archives and are **excluded** from the matched v6 comparison.
The restored initializer matches the original saved binary exactly at the join
samples. The failed hypothesis assertion and its correction are preserved.

Reproduce the new audits with `test_algebraic_chart.py`,
`test_initial_representation.py` (also pass `--legacy-library` pointing to the
preserved qualification-v1 library), and `analyze_chart.py`. Each takes
`--library` and a fresh `--output`; the comparison additionally takes
`--evidence docs/spherical-optimization/evidence`. All runs remain local.

## Boundary, convergence and timestep investigation (v7–v10)

The experimental `--derivative-boundary` policy evolves the endpoint D with
`HyperbolicAuxiliaryRate`, and solves for normal lapse/shift derivatives so the
shared scri kernel gives a fixed clock and the same D trace. Degree-seven
Hermite corrections use seven nodal values and that derivative in the final
three rows. This policy is **rejected**. Its algebraic/clock identities pass
and its Hermite polynomial errors are below 8e-15, but the 128-point 1M run
reaches a pole residual of 469. The leading initial boundary growth rate rises
from 124.02 at N=32 to 248.29 at N=64 (coordinate-time units). Quartering or
doubling the penalty leaves positive growth rates of 240.41 or 86.22 at N=32.
The mode is localized at scri. Smaller RK steps cannot remove this spatial
instability. The plot and saved matrices in `boundary-rejection-v8` and
`semidiscrete-derivative-*-v8` preserve this failure rather than counting its
roundoff clock/D identities as success.

A separate stage-materialization hypothesis was tested and falsified: the
original policy already reconstructs the dependent D before computing auxiliary
advection in **every RK stage**. Moving reconstruction earlier changes the
entire 1M trajectory by exactly zero. The extra implementation was removed;
its run/source and disposition remain in `chart-stage-n128-1M-v9`. A regression
check varies the redundant endpoint D input from +33 to -27 and obtains exactly
the same RHS and RK step. Centered diagnostic D derivatives now also describe
the reconstructed trace. The auxiliary evolution was already using it. This
small diagnostic correction does not change the evolved solution.

`audit_semidiscrete.py` now saves finite-difference Jacobians, their refinement
differences, eigenvalues/eigenmodes, and RK4 versus exact frozen linear-flow
amplification for characteristic targets .1–.5. The state chart removes the
redundant clock/null coordinates and, for the original closure, the dependent D
coordinate. The original-policy N=64 initial and 1M largest real eigenvalues
are 2.343 and 2.155; the most negative eigenvalues are -703.55 and -703.86.
The initial **decaying-eigenspace** RK4 absolute-stability threshold is
C_char=0.60691. Positive growth eigenvalues preclude a contractivity claim for
the full operator. Large singular amplification is reported in a disclosed
local-variable Euclidean norm, not presented as a physical energy estimate.
Jacobian refinement differences are about 3e-11 relative. The matrix exponential
is independently checked against SciPy to 3.9e-15 relative. Early v7 audits,
without source archives, and the v8 original-policy stage-map caveat are
superseded by the archived v9/v10 audits; the caveat was disproved by the
stage-materialization test above.

Reference preparation is now configurable with `--reference-step`. At N=128,
h=.0005, .00025, .000125 gives exterior indicators .00395850545,
.00395852484, .00395852521 at 1M. Thus reference-stencil error is measurable but
is not the dominant exterior error in this control. This is a numerical
refinement check, not a claim of analytic reference jets. The actual continuum
background RHS remains in every evaluation.

At h_reference=.000125, halving and quartering the 128-point timestep gives
maximum exterior field-difference RMS values 2.594e-11 and 1.581e-12. Resolved
component ratios correspond to orders 3.85–4.09; values below the diagnostic
floor are left unassigned in `temporal-convergence-n128-v10`. Temporal error at
this timestep is negligible compared with the spatial error. These are 1M
controls, not temporal qualification of a 10M finalist.

The original 20–40M slicing control was extended to N=512 at 1M. Its common
reference-exterior physical constraint indicator is .00086827, versus .0070074,
.0039585 and .0019897 at 64/128/256. The D compatibility defect, however, is
+3.019 at N=512, versus +2.347 at N=256: decreasing exterior constraints alone
does not qualify the boundary. Spatial field comparisons use exact shared-map
inversion at R_reference/M=25–100. Interpolating total fields is contaminated
by the sharp initial curvature profile; the additional reference-interpolation
comparison differences evolved deviations from the same exact initial fields.
It retains both interpolation estimates. The latter maximum adjacent-pair
field differences are .09688, .05354 and .02378; Lambda/D ratios remain near
first order. No clean high-order spatial claim is made for this family.

The explicit N=64 1M timestep sweep completes at initial C_char=.1 through .6
and fails at .7 (last accepted time .397M, nonpositive chi at scri). Differences
from the .1 trajectory stay below 7.42e-8 in the common exterior through .6.
That short-run result **does not establish a robust timestep**. Ten-M targets
at .1, .3 and .5 fail after 4.628M, 4.520M and 4.442M respectively, with large
scri/constraint errors and nonpositive conformal metric. All failed stages,
last accepted states, source archives and subprocess logs are retained. At
N=128, C_char=.3 completes 10M in 80.49 seconds without floors, but has clock
residual .748 and pole residual .969; it remains unqualified. The shorter
64-point failures cannot be attributed solely to an excessive timestep.

A controlled wider slicing run retains the 20M central physical scale and
changes only the configured slicing end ratio from 2 to 4 (20–80M). With the
same chart, reference-step, analytic normal, separated gauge, ungraded grid,
KO and penalty policies, N=128 at 1M improves the common exterior indicator
from .0039585 to 3.663e-6 at similar runtime (15.51 seconds). Its clock residual
is -1.55e-7 and differential D defect about 5.05e-5. This is a promising
initial-slicing/map change expressed through the shared C++ implementation,
not a frozen background or a claim about boosted/3D stability. Resolution and
10M checks are recorded below as they finish.

New reproductions are provided by `analyze_boundary.py`,
`analyze_convergence.py`, `audit_semidiscrete.py`, `sweep_timestep.py`, and
`analyze_timestep.py`; each refuses to overwrite its output directory. The
convergence and comparison scripts accept explicit case lists. Run manifests
contain complete command configurations; the latest archives also include
the shared Python radial-operator dependency. Some independent local controls
ran concurrently, so these wall times are exploratory observations; a final
fixed-error frontier must use controlled timing conditions and matched duration.

The wider-family 1M checks completed at N=64/128/256 with exterior indicators
4.838e-5, 3.663e-6 and 1.043e-7. The corresponding D defects are -7.65e-3,
5.05e-5 and 5.21e-5: the last pair has reached a compatibility floor rather
than demonstrating further decay. Reference-interpolated field differences
fall from .005073 to .0001826; component ratios range from about 3.65 to 5.83.
Interpolation disagreement remains substantial for the finest curvature
comparison (.0001529 maximum), so the ratios alone do not prove asymptotic
field convergence.

The 128-point wide case also completed 10M at the default penalty timestep,
without floors, in 154.08 seconds. Its exterior indicator grew to .01045,
clock residual to .568, pole residual to 1.578, and differential D defect to
162.43. The horizon mass diagnostic is .28088 rather than 1/3. Consequently
**it is not admitted as a 10M finalist**. The short-time improvement is real,
but it does not survive as a qualified long-time accuracy result. Further
resolution, central-grid allocation and boundary/source analysis are needed.
The comparison in `width-10M-comparison-v10` uses different timestep policies
for the narrow and wide cases and is not a controlled speedup claim.

The single physical central radius remains 20M in both cases; changing the
slicing map consistently changes its compact radius from .64029 to .44017,
and the Cartesian scaling from .10117 to .06955. The conformal/gauge joins
follow those derived radii. Thus this is a controlled **configured slicing/map
family**, not a claim that every compact-coordinate window was held fixed.
The reduced compact allocation to the physical central region is a relevant
next resolution test. The boundary-rejection, temporal-refinement and wide-family resolution plots
were rendered and inspected. No Aurora run, parent-campaign edit or 3D
stability assertion was made.

## Radial allocation, reference tails and Release build (v11–v13)

The areal 20–80M family now has 10M controls at N=64/128/256. The common
reference-exterior constraint indicators are .01673, .01045 and .004930.
At N=256 the horizon mass improves to .333258, but the clock residual remains
.1001 and the differential D defect is -40.12. Lambda/D field differences
still converge only about first order over these resolutions. There is no
admitted 10M candidate.

The dominant exterior error is localized near r=.95, not broadly distributed
through the central plateau. At N=256, peak H and M occur at reference
R/M=62.5 and 68.2; the Lambda/D gradient half-maximum covers only **two grid
points**. The lapse/shift gradient half-maximum covers six. Saved unsmoothed
profiles, gradients and the inspected plot are in
`width4-long-localization-v13`. The preceding v12 analysis directory records a
JSON integer-serialization error, not a simulation failure.

Both signs of the monotone quadratic radial grading are now available.
`--grading -.3` concentrates points toward the puncture chart; `+.3`
concentrates them toward scri. Separate controls either hold the coefficient
kappa fixed or set kappa/dr_scri equal to the uniform-grid value. No state
floors or extra KO were added. At N=128 and 10M:

| Grid / penalty policy | Exterior indicator | Horizon mass | Clock residual magnitude |
|---|---:|---:|---:|
| Uniform |.01045|.28088|.568|
| Central refinement, matched rate |.01228|.34312|.449|
| Outer refinement, matched rate |.007853|.19961|1.110|
| Both ends refined, fixed coefficient |.003941|.33373|.0444|

The two-end option is `--grid-map two-end --grading -.5`, using
r=x+g*x*(1-x)*(1-2*x). Its Jacobian is positive for the accepted |g|<1 range.
Grid/stencil checks reproduce degree-six polynomial derivatives at the
one-sided puncture and scri rows to below 3e-11. The two-end **matched-rate**
control lowers kappa to .33314 and fails near .2065M; a four-times smaller
step fails near .2067M. Its initial semidiscrete operator has a scri-localized
growth eigenvalue 401.74. Retaining kappa=.65104 reduces the largest real
eigenvalue to .6254 and completes 10M, but its differential D defect is still
108.26. Thus equal penalty rates across grids are not sufficient for boundary
stability, and the surviving control is not yet accurate enough to admit.
The fixed/matched coefficient comparisons and failed stages are preserved.

`NativeWormhole::OuterRadialDerivatives` supplies shared C++ derivatives on the
exact S=1 outer plateau, using the inverse-map ODE rather than differencing
quadrature noise. At scri, chi'=chi''=0, alpha'=alpha''=1 and beta'=-1, beta''=0
in the laboratory's total-field representation. `--reference-tail scri`
applies just the endpoint limits; `outer` also uses the analytic differentiated
map in the outer plateau. Neither option changes evolved equations or resets
constraints. Independent 70-digit integration/differentiation agrees with
outer derivatives to 5.69e-8 at r=.999 and much better farther inside; the
remaining error comes from the double-precision map value and cancellation.
The 128-point 1M D defect decreases from 5.05e-5 to 2.14e-6 with exact scri
jets, while the exterior indicator is unchanged. At 10M the large defect is
unchanged to useful accuracy. Reference differentiation explains a small
short-time floor, not the dominant long-time problem.

Full 26-field point-symbol checks now include radial, transverse and oblique
directions on sampled evolved states through 10M. The inspected wider N=128
run has no numerically defective eigenvalue clusters at those samples, and
imaginary parts remain below 7e-16. Eigenspace condition numbers reach about
1.8e4. This is not uniform endpoint hyperbolicity, a nonspherical evolution
test, or evidence of boosted/3D stability.

The CMake default `Release` assignment did not replace an already-created
empty cache entry. Both earlier build directories therefore omitted optimizer
flags. The corrected default preserves explicit user build choices and selects
Release when the cache is empty. A separate `build-spherical-v12` uses `-O3`,
`-DNDEBUG`, and the same `-ffp-contract=off`. On 257 off-constraint jets its
normalized RHS difference from the old library is exactly zero; initial data
differ by at most 1.11e-16. The matched uniform N=128 10M run drops from 154.08
to 29.24 seconds, with maximum final field difference 6.81e-10. This observed
5.27x change is **compiler optimization**, not a formulation or GPU speedup.
It is kept separate from physics comparisons; overlapping independent controls
also make these exploratory timing observations. Both binaries, flags and
caches are preserved. New run manifests capture build metadata, and plots
mixing binary identities explicitly prohibit ranking runtimes.

The handoff's compact-coordinate slicing construction is now implemented in
shared C++, preserving the central compact scale while spreading the outer
feature. The implementation, independent verification and new 10M evidence are
in [COMPACT-PROFILE.md](COMPACT-PROFILE.md). At N128 it improves the matched
20–40M anchor's common-exterior indicator by 51.5x at 10M. Differential D
compatibility remains unresolved, so this is a promising spherical candidate,
not an admitted stable formulation or a 3D/boosted result.

The next reference-variable controls are in
[REFERENCE-VARIABLES.md](REFERENCE-VARIABLES.md). Shared Cartesian derivative
rules for chi/r² let the N512 run complete 10M without floors, with exterior
indicator 1.38e-6. The chi/r⁴ control fails and is preserved. Smooth gauge and
constraint pulse tests now include characteristic-direction probes and a 20M
check; late gauge-response growth and the nonconverging D boundary defect
remain explicit admission failures.
