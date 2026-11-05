# Local spherical optimization laboratory

Branch: `codex/spherical-gauge-optimization`, based on `718ad1fdd` of
`codex/refined-convergence`. This directory is a separate local experiment.
The parent campaign, frozen Aurora snapshots and remote jobs are untouched.
**Work in progress: no candidate is admitted, no stability/cost improvement is
established, and no spherical result establishes boosted or 3D stability.**

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
cmake -S tst/spherical_optimization -B build-spherical \
  -DKOKKOS_SOURCE=/path/to/local/kokkos
cmake --build build-spherical -j 4
ctest --test-dir build-spherical --output-on-failure
python3 -m venv build-spherical/python
build-spherical/python/bin/pip install numpy scipy matplotlib
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
