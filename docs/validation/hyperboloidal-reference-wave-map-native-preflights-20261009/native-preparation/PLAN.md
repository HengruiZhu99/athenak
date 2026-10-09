# Held actual-native reference-wave-map control

Status: source-only. No compiler, AthenaK process, spectrum, operator assembly,
propagation, or black-hole evolution is released by this document. Root must
review these sources/recipes and the saved finite-amplitude RHS readback before
issuing an exact build/run authorization. All production files stay byte-identical
to implementation `27c19d20696ea6dd4704032c51dfd026218f64f2`. Record the later
launch HEAD independently of that compiled implementation.

## What is being tested

The private candidate replaces the four native lapse/shift RHS rows by the frozen
physical-reference wave-map helper `reference_wave_map.hpp` (SHA
`56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28`).
The condition is physical `g^{bc}(Gamma[g]^a_bc-Gamma[ghat]^a_bc)+2 Z^a=0`,
with stored physical `P=K-2Theta` and the actual integrated Lambda containing Z.
This is a global harmonic-core diagnostic control, not a moving-puncture gauge
or a choice for the later wormhole-to-trumpet black-hole transition.

The C0 geometric P/Theta/chi/metric/A/Lambda equations, kappa input 10, kappa2=0,
all initializers, final-stage-only algebraic projection, fourth-order native
Dx/Dxx/Dxy, native upwind advection, sixth-order KO 0.1, and existing symmetric
degree-2 spherical continuation are unchanged. No C1, spatial-norm eta6,
preferred-source projection, null feedback, damping profile, Q replacement,
Omega floor or imposed Theta falloff is present. Existing C0 geometric RHS
subtracts only its analytic Minkowski reference floating-point residual, exactly
as the production patch already does. The new gauge uses analytic factored
stationarity rather than a computed full-RHS counterterm. The 48-point exact-flat
finite-amplitude gate tested the raw geometric RHS, before that native C0
roundoff subtraction; the native reference preflight must test this distinction.
Prepare still interpolates each deviation component from strict interior donors
and restores the full ghost values. Only active cells receive the algebraic
projector; ghost metrics are not projected, and no boundary-jet/constraint
compatibility enforcement is inferred from this unchanged closure.

The preserved baselines are (a) all prior production evidence and (b) a matched
fresh C0 physical-P/source-off native control with the same new pulse, geometry,
grid, output and timestep caps. Prior span2.1/.1/.02/width.5 results are context,
not the matched comparator for this span2.2/.2/.1/width.35 control.

## Exact native seam and build isolation

`prepare_build.py --prepare-only` never compiles or evolves. It makes three fresh
recipes: `wave-map`, `c0`, `wave-map-half`. Each starts from the exact public
CartesianPatch header and the pinned original Release object/link recipe.

The wave-map overlay changes only one include and one evolution call-site plus
an explicit `xyz[3]` local at that call. `LayerPoint` has radius but no orientation;
the coordinates come from `g.first+i*g.h` used in the very same `ref.At` call.
`ResearchNativeWaveMapGauge` calls `ReferenceConnection(p,xyz)`, `Gauge`, then
`rwm::Assemble`. This last function adds regular + pole/Omega for alpha AND all
three beta rows. Production `AssembleGaugeInterior`, which adds only the alpha
pole, is not called on these candidate parts; there is no second beta addition.
An exact inverse patch must recover the public header byte-for-byte.

No function-name macro or production LayerPoint change is used. The old forced
spatial-norm injection is removed completely from each selected compile command.
The matched C0 recipe compiles the public header byte-identically, also without
the old injection. Dependency closure from the original native `.o.d` files
selects exactly six CartesianPatch-consuming source files; those six objects are
rebuilt and replace their original link entries. All 176 other linked native
objects and libraries are reused and rehashed before/after. Compiler flags stay
the pinned CPU Serial, double, C++17, arm64 Release `-O3 -DNDEBUG` flags. Every
compile command/cwd, generated depfile, compiler version/hash, private source,
reused object/library, executable and log must be saved before/after execution.
The runtime's existing one-block/uniform-3D/ng3/Serial/double/no-MPI/no-matter/no-
floors/rk3 restrictions remain active. All planned inputs set mass=0.

The six-TU closure and exact recipe inputs are saved by the preparation receipt.
If a dependency or hash differs, stop and preserve that mechanical failure;
do not silently reconfigure the base tree or reuse a drifting executable.

## Native pulse and fixed grids

Use the production initializer without changing its formula:

`s=(1-r^2)^4 exp(-r^2/width^2)` for active `r<1`;
`delta alpha=A*s*(1+.2*x+.3*y*z)`;
`delta beta=B*s*(1+.3*y*z,.2*x,.1*x*y)`.

It is smooth at the origin and vanishes to fourth order at scri. It is not a
compact cutoff inside scri, and its zero extension is not claimed C-infinity.
There is no already-available compact-support switch. Angular=true is fixed,
so neither lapse nor shift pulse is spherically symmetric.

All cases use S=1, curvature a=.5, geometric layer .05--.95, existing gauge-layer
parameters .45--.85 (used only for the retained conservative timestep bound in
the wave-map run), ghost degree2/symmetric, kappa10, KO.1, ng3, RK3, and the same
uniform cube [-1.1,1.1]^3 at N16/N24/N32. At N16 span2.1 is rejected by the
production planner: the physical cell center .984375 is inside the unit sphere,
so full ball+halo coverage fails. Span2.2 has physical end center1.03125 and
satisfies that unchanged admission check. All resolutions share span2.2.

Input flags physical_trace_lapse=true/preferred_source=false define the matched
C0 comparator. The direct wave-map call deliberately ignores those gauge-source
flags; they do not activate a hidden lapse blend. The stored P and live fields
are identical conventions in both runs. No runtime production option is added.

| case | lapse A | shift B | width | angular | initial tests |
|---|---:|---:|---:|---|---|
| reference | 0 | 0 | .35 | true | N16/24/32, t=.05 |
| small control | .02 | .01 | .35 | true | N24, t=.02 then t=2 |
| large pulse | .2 | .1 | .35 | true | N16/24/32, t=.02 then t=2 |

Run reference and .02 preflights before longer cases. Matched C0 uses the same
reference/large-pulse matrix; the small control is secondary amplitude evidence,
not a substitute for the large pulse. A negative or inconclusive result is
retained without changing amplitudes, geometry, ghost policy or thresholds.

## Actual timestep control

The unchanged native cap is
`dt=min(.025*h/max_speed, .03*Omega_min)` after Mesh multiplies by input CFL.
The hyperboloidal function first divides by that same input CFL, so .1 versus
.05 in the input alone would not be a half-step experiment. Keep input CFL=.1
for all files and record every actual dt/restart-header dt, including endpoint
clipping and the driver's maximum factor-two growth rule.

The retained speed is `max_i(|beta_i|+sqrt((alpha^2+2(1-W)alpha)*chi*gInv_ii))`.
For admitted alpha>0 it bounds the wave-map light speed
`|beta_i|+alpha*sqrt(chi*gInv_ii)` from the actual full20 principal gate. Keeping
this larger bound preserves the baseline timestep machinery. It does not prove
stability for singular lower-order terms or the spherical boundary closure.
This bound is scoped to the constrained20 principal system; it does not certify
the unrestricted22 intermediate-stage algebraic-normal dynamics.

The N24 `wave-map-half` build changes only the private copy of z4c_newdt.cpp by
`dtnew *= .5` inside the hyperboloidal branch after computing the original cap.
This halves both spatial and pole caps. It is among the same six selected TUs;
the exact source diff and command are saved separately. It is a timestep-only
control, not a change to gauge or principal equations. No claim of an identical
pointwise dt ratio after the two nonlinear states diverge is made; each state
has the same cap formula times the fixed .5 factor.

## Fixed outputs, arithmetic gates and diagnostic decisions

Save binary64 restart arrays and native history at t=0 and every .025 through
t2 (preflights every .005). Keep separate output directories and exact input
bytes, commands, executable hashes, console dt histories and completion codes.
Production visualization `bin` output may be lower precision; restart arrays
are authoritative for small residuals/field comparisons. Save constraints and
ADM outputs as well if budget permits; no output may overwrite a prior case.

Before interpreting evolution, verify the compiled native seam on actual loaded
reference and six declared finite input cells: complete actual gauge equals
the frozen helper and both alpha/beta poles enter once (2e-12 scaled). The
production upwind correction is applied afterward to deviations exactly once
and must be separated from this centered-jet seam comparison. For all source/
binary assertions preserve failed receipts and never retry with relaxed tolerances.

Reference preflight PASS requires all active full fields finite, alpha/chi>0,
SPD metric, actual det/trace residuals <=1e-11, max reference drift <=1e-10,
and RMS H/M/Z/Theta <=1e-9 at every saved sample. This is an arithmetic/native
fixed-point gate, not evidence of finite-pulse stability. Compare native C0
geometry subtraction with the raw 48-point prerequisite explicitly.

All finite-pulse runs require completion, all active fields finite, alpha/chi>0,
SPD (record minimum eigenvalue), det/trace residuals <=1e-10, and all diagnostics
finite at every saved binary64 sample. Record per-component deviations, extrema,
H, conformal-norm M/Z, physical Theta, radial shells, maxima and their locations.
The production pole history is **geometric C0 only**; it does not include the
new gauge poles. Add a separately labeled values-only gauge-pole readback from
the same snapshots using the helper, never describe that as a full live-source
or derivative check. No floor or artificial field cap is allowed to make a pass.

First assess t2 on the three large-pulse resolutions, N24-half, and matched C0.
Report all H/M/Z/Theta individually. For a useful improvement at N24 require
each final H/M/Z <= matched C0 and at least two reduced by >=20%; distinguish
this relative control verdict from acceptance. For continuation to t6, also
require N32 final H/M/Z <= .8 times corresponding N24 values (a numerical
resolution trend, not a measured order), the N24 half-step final H/M/Z changes
<=10% relative to N24, and no component RMS more than doubles between t1 and
t2. Zero denominators use absolute1e-10 comparisons. A failed condition stops
automatic t6/t12 extension and is reported, without claiming a PDE instability.

If all these fixed continuation gates pass, N24/N32/half may continue freshly
to t6, with same output cadence and finite guards. Continue to t12 only if
the last-unit-window H/M/Z/Theta maximum at t6 is <=1.25 times its previous-unit
window, the same N32/N24 and half-step controls remain satisfied, and root reviews
the saved diagnostics. A t2 improvement alone is not long-time acceptance.
Successful long runs must demonstrate bounded constraints and a reproducible
resolution trend, not merely positive fields. Later BH acceptance still requires
an independently justified inner wormhole-to-trumpet gauge with the Minkowski
hyperboloidal reference retained; this global harmonic diagnostic does not supply it.

## Known scientific limits

The exact rational harmonic full20 principal proof and 792 finite actual symbol
checks do not certify finite-k lower-order or boundary stability. The 48-point
finite-amplitude exactly-flat oracle establishes source/ADM convention binding,
not every nonlinear Einstein solution or an exact-scri closure. The current
interior grid always has positive Omega; no value is assembled at scri. C0 is
not silently converted into fully covariant Z4, and no subsidiary energy or
nonlinear Theta asymptotic assumption is introduced. These are actual native
arrays/RK3 controls, with whatever outcome the unchanged boundary continuation
produces; no separate spherical evolution is a replacement.
