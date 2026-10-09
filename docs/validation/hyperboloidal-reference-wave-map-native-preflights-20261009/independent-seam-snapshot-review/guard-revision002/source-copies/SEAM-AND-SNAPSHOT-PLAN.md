# Fixed source-only native seam and snapshot readback

This is additive to the reviewed PLAN and already prepared/compiled recipes;
their bytes stay unchanged. Native evolution remains held. Root must separately
review/release the probe source/recipe and analyzer before any scientific call.
Subsequent stages record current launch HEAD `e654ceb0602c5fc47d8e0c600aedda660a96c92f`
or the actual later HEAD, independently of compiled production27 and the earlier
three build receipts' launch HEAD.

`native_seam_and_snapshot.cpp` has only two modes. Neither performs a time step,
operator assembly, spectral solve or propagation. The held probe command is
prepared by `prepare_probe_recipe.py` from the already reviewed wave-map compile
flags/include resolution and the same four immutable Kokkos libraries. It includes
the exact private CartesianPatch overlay and frozen helper. All source/compiler
dependencies, command/executable and stdout/stderr must be pinned before execution.

## Declared seam cells and states

N24, cube[-1.1,1.1]^3, ng3, h=2.2/24, stored first coordinate
`-1.1+(0.5-3)*h`, and physical reference a=.5, geometry.05--.95, S1, kappa10,
degree2 symmetric continuation, KO.1. Indices below are **full stored i,j,k**,
not active-box offsets. The JSON `seam-cases.json` pins their binary64 xyz.

| full i,j,k | approximate radius | scope |
|---|---:|---|
|14,14,14|.07939|inner geometric transition, not exact core|
|18,15,14|.32731|geometric transition|
|20,17,14|.55570|geometric/gauge transition|
|22,18,16|.77104|geometric/gauge transition|
|24,17,16|.91092|harmonic outer gauge region, geometric transition|
|24,19,16|.97335|outer exact CMC region|

Each cell is tested at initial time t=0 in exactly three array states: reference
(A,B)=(0,0), small(.02,.01), and large(.2,.1), width.35/angular=true. Thus18
fixed rows, no adaptive state/cell selection. A standalone array constructor uses
the exact source-pinned production pulse formula; it does not invoke the full
Mesh/Z4c initializer. That separate runtime binding is checked on the actual t0
restart before interpreting any evolution.

The probe calls the actual private CartesianPatch.RHS on the complete arrays.
After its Prepare, it independently reloads the same deviation stencil with
native LoadMeshJet<3> and analytic reference jets. Its manual comparator uses
the frozen helper's regular/pole parts but **does not call** the tested wrapper
or rwm::Assemble: it explicitly forms regular+pole/Omega for all four gauge
components, reconstructs the unchanged C0 geometric reference-residual subtraction,
adds the one native `(Lx-Dx)` deviation correction using full beta, and adds the
one KO term. Compare all22 stored RHS fields, scaled by max(1,abs(expected),
abs(actual)), threshold2e-12. Reference actual-RHS max must be <=1e-10.
At least one finite row must witness >1e-8 absolute error if beta poles were
omitted or an alpha pole duplicated; these are arithmetic negative controls,
not alternative evolved sources. Save actual/manual22, gauge regular/pole4,
coordinates, Omega, alpha/beta values and per-row errors. No gauge f0 subtraction
is introduced. This tests the native header/array seam, not all task scheduling.

## Snapshot transport and diagnostics

The source-only Python analyzer requires exact per-case launch authorization,
input/executable/build-receipt hashes and a separately pinned successful probe.
The previously validated RST reader/ABI are reused byte-for-byte and rehashed.
The reader rejects different ABI, fields, number of blocks, refinement, matter,
trackers, incomplete payloads and trailing data. All25 variables, ghost-inclusive,
are consumed as little-endian binary64 LayoutRight `[1,25,k,j,i]`. The exact raw
payload is passed to `--snapshot N` with its hash and byte count retained.

Recompute the strict spherical active mask using the stored grid geometry;
check active alpha/chi positive and every component finite. Reconstruct each
symmetric conformal metric from its six stored components and require positive
finite eigenvalues. Those3x3 eigenvalues are field-admission diagnostics, not a
PDE/operator spectrum. All inactive/ghost values are retained in the original
RST but are not silently added to active physics norms.

The compiled read-only probe reconstructs the exact same reference and Prepare
closure, loads the actual native centered jet and calls EvolvedConstraints.
It reports active-node RMS physical H, conformal-contracted M/Z and physical
Theta, maxima/xyz, radial-bin budgets, shell r>=.9, det/trace residuals,
alpha/chi minima and all25 component deviations. No continuum interpolation
or guessed reference jets replace the native operators. Recomputed global
H/M/Z/Theta RMS must match the same-time native history within2e-11 scaled;
all outputs must be finite. The reference <=1e-10 drift/<=1e-9 constraints and
finite-pulse det/trace<=1e-10 gates are exactly the original PLAN thresholds.
Initial actual runtime values must match the declared production initializer
formula in every active field within2e-13; initial constraints<=1e-9.

The separately named shell wave-gauge pole numerators are values-only diagnostics.
Their source helper also computes regular values internally, but no full 4D
source/derivative check is inferred from this readback. Production history's
geometric-only pole column remains unchanged and separately labeled.

The analyzer consumes a completed native launch receipt with exact input path/
hash, executable path/hash, build receipt path/hash, mode, command/cwd, output
directory, returncode and run-log hash. Source-only `launch-schema.json` specifies
these fields. A scientific failure or incomplete case is preserved separately;
it is not a completed snapshot PASS. Every called probe input metadata, stdout,
stderr, returncode and summary is retained. Source/analyzer/recipe hashes are
part of authorization and cannot drift mid-case. Large raw arrays remain local
with hashes; future compact archives must mark all NPZ/NPY or >1MiB cases payloads
as large_payload even if a particular array happens to be small.

## Additive scope and timestep/cost clarification

The nearest active radii for span2.2 N16/24/32 are approximately .11907849,
.07938566 and .05953925. None samples the exact r<=.05 Cauchy core. The earlier
independent local17-witness core gate covers that distinct regime; no nearest
native cell is relabeled as core. The minimum-Omega pole caps from the actual
outer grid imply at least about24735/30721/17196 steps to t2 before any stricter
speed cap or clipping. Preflights and seam admission therefore precede long runs.

The original PLAN's request to record every actual dt cannot be met in full
precision from the unchanged console: Driver prints6digits. Preserve every
console cycle/time/dt as printed, plus fullprecision history dt and restart
header dt at saved samples. RST header dt is current pm.dt when output is written,
not unconditionally the last completed step; history has its own same-stage
semantics. Recompute the live cap from saved state as a separate diagnostic,
not a false equality to a prior-step dt. Actual half-cap behavior is established
by the pinned source diff and the saved dt controls, not an input CFL claim.
