# HiSpID initial-data integration

October 3 completed Gamma10 damping diagnosis: further runs are held under
the latest preservation instruction; the performance campaign remains stopped.
The exact isolated seed's L96/ntheta98 trial uses alpha=.02, initial scale1.02,
a200-iteration cap and the unchanged1e-7 expansion RMS criterion. It times
out at900s. All45 complete surface-integral snapshots retain positive minimum
radii; expansion RMS falls from.443830 to.0790216. This supports improved
flow stability over the recorded interval, without establishing convergence.
Its area remains an attempt diagnostic, not a qualified horizon mass.
The import roundtrip error is4.65942e-16; terminal time/cycle verification
remains false after timeout. A single angular order cannot satisfy the
unchanged three-order refinement rule even if a worker were to converge.
Full trace, frozen sources, original failed receipt and local hash verification
are retained in `hispid-gamma10-damped-20261003/`. Allocation59296178 completed
and is no longer queued. No new run was started during preservation.

October 3 retained Gamma10 solver retry: the separate restart200/budget4800
trial from the failed128×256×8 checkpoint reaches the original1e-14 internal
tolerance in2 additional Newton/574 Krylov steps, while independent near-hole
momentum RMS remains0.217375. The binary remains physically unvalidated.
Its fresh diagnostic checkpoint passes2454-point separate-process producer/CPU
sampling, including metric derivatives, with zero measured differences.
AthenaK import-only passes at ADM/Z4c error4.59847e-16 with time/cycle zero and
no finder constructed. The exact receipt, migration proof, input and run log
are in `hispid-extreme-gmres200-20261003/`. They establish interchange only.
The native branch retains all raw arrays, frozen sources and the separate
incomplete HS99UU attempt in `validation/extreme_trial_retention_20261003`.
Allocation59293910 has been released. No further numerical work was launched
during preservation; existing performance results and failed flags remain
unchanged under the HUMAN OVERRIDE.

October 3 completed extreme imports: both coarse chi=.99 and Gamma10 binary
checkpoints retain their diagnostic acceptance labels. Fresh producer/CPU
sampler proofs compare2454 points per checkpoint and every physical/conformal
field and metric derivative with zero difference. The spin binary imports
with ADM/Z4c roundtrip error4.633e-16, but its first component finder attempt
fails at expansion RMS1.941e-5. The Gamma binary passes the separate
`check_hispid_import.py` check with error4.809e-16, time/cycle zero, and no
finder constructed. This driver shares the checkpoint/migration/import
helpers and binds inputs, executable, source, libraries and retained log;
it establishes interchange only. Both binary constraint acceptance flags
remain failed and no enclosure is established.

Exact isolated Kerr chi=.99 passes the L8/12/16 controls, with area
28.6781506783, Christodoulou mass1, coordinate-axial chi error below1.4e-12
and expansion RMS below1e-7.
Exact Gamma10 factorized L64/96 searches fail invalid-surface checks and
L160 times out at900s. No measured Gamma10 horizon mass or calibrated
binary d/Mirr=50 is claimed. Further performance checks remain stopped;
original successful and failed measurements are preserved.
The unchanged import/finder receipts and logs are in
`hispid-extreme-20261003/`; the native branch retains the complete raw-data
manifest and sampler arrays in `validation/extreme_physics_20261003`.

October 3 physical follow-up: the user stopped further performance work while
preserving the existing report and failures. The new Perlmutter Serial
consumer builds with a separately verified pure CPU sampler. Fresh exact
Kerr chi=.99 and Gamma10 checkpoint sampler comparisons agree at 1512 points
per case, including physical metric gradients. The harmonic cache component
test passes exhaustive L8/16 checks and sampled L160 checks; its bound receipt
and log are in `hispid-fastflow-cache-components-20261003.json` and the sibling
`.log`. Two displaced speed=.885 finder comparison attempts timed out at900s;
their successful import does not qualify full finder equivalence. Centered
target horizon measurements are separate. No new solved binary or measured
Gamma10 horizon is claimed by these component checks.

This isolated branch starts at PR790 head
`22baa243970fa1880b2bbc48e88a590069d55e47` on
`HengruiZhu99/athenak:project/z4c_overhaul`. It adds the problem generator
`z4c/hispid`; no shared installation or original checkout is changed.

Build the external isolated native backend first, then configure a fresh
AthenaK build directory with explicit paths:

```sh
cmake -S . -B build-hispid -DPROBLEM=z4c/hispid \
  -DHISPID_ROOT=/absolute/path/to/TwoPuncturesC \
  -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_SERIAL=ON
cmake --build build-hispid -j1
```

Double precision is required. CMake resolves the header under HISPID_ROOT
and the library under its build-hispid directory. For a separately preserved
verified producer, set -DHISPID_LIBRARY_DIR=/absolute/path/to/producer-build.
It prints both paths and records the library SHA256. Reconfigure
and rebuild after changing the external library. The checkpoint source SHA
must match that consumer SHA by default. Explicit library migration is for
separately validated builds; a matching mathematical parameterization alone
does not validate a new compiler, source version or backend.

The native `examples/export_athenak.py` exports either a bound solved case or
an exact seed control. Its portable text format records all configuration
fields explicitly, seventeen digit unknowns, source SHA and acceptance label;
it never dumps native structure padding. The reader rejects missing/extra
fields, invalid counts, nonfinite data, incompatible versions and excessive
allocation budgets. Diagnostic cases need an explicit input flag. A solved
case's aggregate acceptance does not certify every coarse record.

Use `tst/inputs/hispid.athinput`, overriding `problem/hispid_filename` with an
absolute checkpoint path and `problem/hispid_source_sha256` with its source
SHA. The pgen loads physical gamma_ij/K_ij into all active and ghost cells,
converts ADM to Z4c and back, and checks every component. It does not multiply
the imported physical metric by another conformal factor. It sets the usual
precollapsed lapse after conversion. Exact punctures are outside the sampler
domain; there is no silent fill or radius-floor substitution.

`hispid_initial_horizons=true` calls AthenaK FastFlow at time/cycle zero.
Configure one surface per active hole and time windows containing zero.
`hispid_direct_horizon_geometry=true` supplies native metric, extrinsic
curvature and analytic metric first derivatives at angular nodes. Rank zero
owns the geometry, and global coverage must be exactly one per point. The
callback and seed-shape guess are cleared before the context is destroyed.
This tests the imported dataset using AthenaK's surface finder; mesh-resolved
finder accuracy is a separate check (`hispid_direct_horizon_geometry=false`).
The existing mesh derivative path still needs its ghost-derivative defect
repaired before precise block-boundary horizon claims.

Finder acceptance requires a positive finite surface and expansion RMS in
addition to mass stabilization. The upstream `hrms` summary column remains
mean square expansion; the pgen reports its square root explicitly. Two
half-open angular reductions now include the last point. Failed attempts
clear stale horizon properties and retain their last area/RMS diagnostics.
FastFlow scalar harmonics use a normalized Legendre recurrence; general
spin-weighted harmonics are unchanged. The new recurrence is tested on GL
nodes (which exclude poles) by analytic low modes, addition theorem and
independent finite differences through high orders.

`hispid_mesh_constraints=true` writes a JSON diagnostic using AthenaK's ADM
constraint calculation, separately for g=1, g<1 and a stencil-safe outer
region. RMS uses coordinate cell volumes. The outer exclusion is
max(requested minimum radius, g_max + conservative stencil halfwidth) about
each active puncture. Keep the requested minimum fixed above every grid's
guard when comparing a common refinement region. The actual FD stencil is
recorded. These diagnostics do not automatically accept coarse mesh data.

The standalone `tst/test_suite/z4c/check_hispid_controls.py` uses exported
analytic checkpoints and fresh run directories. It checks cycle zero, exact
area/spin controls, boosted angular convergence, and an independent sampled
shape reconstruction. Low angular orders are diagnostic and retain failed
strict flags. Binary attenuation enclosure requires a separate surface/center
and refinement bound; isolated seed horizons do not establish it. The current
coordinate-rotation spin integral is not an approximate-Killing-vector spin
for generic boosted or distorted horizons.

Recorded serial validation is in `hispid-validation.json`, including failed
coarse angular flags and the unstable combined alpha1 run. Exact isolated
spin chi=.95 and boost v=.885 controls pass separately and together. With
lmax48, the combined seed needs alpha=.2; expansion RMS is9.92e-8 and
independently sampled relative shape error3.35e-7. Its area agrees with
8*pi*(1+sqrt(1-chi^2)) to1.8e-14. Increasing ntheta50 to74 at fixed lmax
changes area by8.4e-15 relative and expansion RMS by3.6e-12. These are seed
rest-spin and lab-speed targets; the boosted coordinate spin integral is
not compared to rest spin.

The mesh constraint runner `check_hispid_mesh.py` uses16³,32³,64³ grids in
the fixed cube[-2,2]³ outside radius1.8. At64³, combined-seed H/M RMS are
6.13e-7/2.23e-7 and maxima8.86e-6/2.28e-6; all four exact seed cases improve
with resolution and pass the declared mesh gate. This validates field
import and AthenaK's outer mesh constraints, separately from the direct
native-geometry horizon checks. No evolution steps are taken.

The current regular-basis consumer compares the checkpoint basis token with
`HiSpID_unknown_parameterization()` from the explicitly linked backend, so
legacy nodal-V arrays cannot be interpreted as modal-P data. Its freshly
rebuilt exact-seed controls are recorded in `hispid-current-controls.json`
(native SHA9cbf1108…, executable SHAeca603e9…). Schwarzschild, chi=.95 Kerr,
v=.885 boosted Schwarzschild and their combined Kerr seed pass again, with
zero evolution steps. Fresh16³/32³/64³ mesh constraint controls also pass for allfour exact
seeds, recorded in the same current evidence file. The combined64³
outer H/M RMS are6.13e-7/2.23e-7. Historical quadrature refinements retain
their earlier fingerprints in `hispid-validation.json`. The current moderate binary has subsequently passed the preliminary
constraint/charge/covariance sequence and its direct-geometry initial-time
horizon/enclosure checks; stronger exterior accuracy remains failed.

The separately built consumer uses native producer SHA126300dc… via
HISPID_LIBRARY_DIR=.../TwoPuncturesC/build-hispid-budget.
`tst/test_suite/z4c/check_hispid_binary.py` checks both components of the
128×256×28 moderate checkpoint at lmax8/12/16, followed by fixed-lmax16
quadrature ntheta32→48. Its direct native geometry and imported mesh
round trip are separate from a mesh-resolved finder check. Both finest
surfaces pass expansion RMS1e-7:9.38e-8 and6.09e-8 at ntheta48. Relative
area changes are<7e-12 under fixed-order quadrature refinement.

The real orthonormal harmonic coefficients give continuous radius bounds
by the addition theorem. Subtracting the actual17-digit finder-center
offset, inner_max and an observed refinement allowance gives enclosure
margins.176225M/.115817M. Continuous upper bounds also certify distinct
components, with separation margin5.555M. This encloses the g/operator
modified balls on the retained surfaces; the refinement allowance is
empirical, and noncompact f/F tails still require exterior constraints.
The coordinate rotation integral does not provide a generic AKV spin.
Full source, checkpoint/executable hashes, coefficients, failed coarse
flags and commands are in `hispid-moderate-binary.json`. These checks take
zero evolution steps and do not promote the failed stronger binary gate.

The revised local spin95 binary checkpoint has also been loaded and measured.
Its native grid is128×256×24, with equal seed masses.5 at x=±6, spins
(0,0,.2375), zero boost, omega1/power4 and actual correction operators.
It carries the explicit diagnostic label because independent physical
constraints still fail. `--allow-diagnostic` enables this measurement without
promoting its physical acceptance.

Both lmax12,ntheta24 surfaces are found with expansion RMS4.77e-6 under the
coarse1e-5 measurement tolerance. Each has area8.3135384, Christodoulou
mass.5006533, irreducible mass.4066849, coordinate spin magnitude.2375000
and coordinate chi.9475223. Their center coordinates are exactly(±6,0,0).
The mesh import/ADM-to-Z4c round trip error is4.1089e-16. This is an
initial-time direct-native-geometry check, with zero evolution steps.

The stricter lmax16,ntheta32 attempt stops at RMS2.3512e-7, above1e-7.
Its angular standard deviation levels near2.32e-7 while the signed mean
continues decreasing. Fixed-lmax quadrature and higher-lmax checks remain
pending. The strict horizon/enclosure flags remain false. The coarse retained
surfaces have continuous inner-ball margins.013955M, but this alone does not
certify the stricter refined surface or exterior vacuum accuracy. The spin
integral uses coordinate rotations; a generic AKV spin is unmeasured.

`check_hispid_binary.py` accepts explicit positive `--flow-alpha`,
`--guess-scale` and `--flow-iterations` controls. All output directories must
be new. Alpha.2 stabilizes this mass.5/high-spin flow; the failed alpha1
cycle is retained. Using the measured mean-radius guess instead of1.05 times
the seed radius reduces the matched coarse search213.69s→44.39s, with relative
area difference9.4e-13. Geometry and acceptance thresholds are unchanged.
Verbose FastFlow output now reports RMS every25 iterations as comment lines,
keeping the existing numeric columns. Failed attempts retain last area/RMS.

`hispid-spin95-binary.json` contains the checkpoint/source/executable hashes,
commands, coefficients, measured properties, resources and all earlier
interrupted attempts. The logging consumer102abcb0… links the same
126300dc… producer. To replay the bounded diagnostic, use the separately
built serial executable and its explicit checkpoint:

```sh
python tst/test_suite/z4c/check_hispid_binary.py \
  --executable /absolute/path/build-hispid-horizon-progress/src/athena \
  --checkpoint /absolute/path/build-hispid-binary/data/spin95-local128.hispid \
  --allow-diagnostic --flow-alpha .2 --guess-scale .97901669168957195 \
  --flow-iterations 100 --levels 8,12,16 --timeout 1800 \
  --output /absolute/path/fresh-spin95-diagnostic
```

The diagnostic returns1 for the retained strict failure. The reported
coarse property measurements and successful import remain available.

The October3 consumer preparation preserves checkpoint memory budgets through
the native65536MiB maximum, while the sampler independently checks its actual
allocation estimate. Dynamic builds verify the actual HiSpID and puncture
symbol images and log their canonical paths. An explicit `--migration-proof`
must bind a separate-process producer/pure-CPU-consumer field and derivative
witness, exact basis/maps, checkpoint SHA and both dependency inventories.
That witness transfers sampling data only; physical acceptance stays separate.
Executable, input, checkpoint, proof and image hashes are rechecked around
every retained horizon worker.

`check_hispid_binary.py --domain-half-width` records a domain containing both
holes and all trial surfaces. `--common --common-radius` performs a separate
one-finder search about its configured center. Component searches retain their
own masses/spins. Common-surface enclosure checks subtract each hole's center
offset and modified-ball radius from a continuous harmonic lower bound, then
subtract the observed refinement allowance. Inconclusive contracted bounds
remain failed, and a failed common search does not prove absence.

A fresh Serial consumer builds successfully with one compiler process; typed
checkpoint and synthetic sampler-proof metadata tests pass. Actual migrated
sampler, new import and common-horizon controls are still pending. No new
chi=.99/Gamma10 binary has been qualified by this preparation.

The existing exact-control and fixed-order quadrature drivers now recognize
separate `kerr99` and `gamma10` checkpoints. Each requires a fresh
separate-process producer/pure-reference-consumer sampler proof in its manifest
entry's `migration_proof`. The case name is checked against the actual mass,
spin, boost, zero corrections and unmodified single-seed configuration.
Every refinement row must preserve import/provenance and time/cycle zero,
including coarse diagnostic surfaces. A fine success cannot waive a failed
coarse import. Quadrature requires the qualified case-specific group.
Timeout logs and unqualified rows are preserved before subsequent validation.

`--consumer-memory-mib` screens the fifteen Serial harmonic tables and an
explicit1GiB allowance; it is not a measured process peak or a complete
allocation guarantee. High-order shape checks use the independently provided
[SciPy spherical harmonic oracle](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y.html),
with polar/azimuthal conventions matching global z, order-dependent displaced
samples and cardinal points. Lower orders keep the independent Legendre
polynomial oracle. These are sampled shape checks, without continuous shape
certification. Their numerical execution remains pending.

For the unit nonspinning Gamma10 control, the horizon's irreducible and
Christodoulou masses are1 while isolated ADM energy is10. For Kerr99 the
expected area is8*pi*(1+sqrt(1-.99^2)) and coordinate Sz=.99. Seed inputs,
measured horizon properties and global ADM charges remain distinct outputs.

FastFlow has an opt-in `fastflow/factorized_harmonics=true` cache for double
precision. The default dense tables, harmonic recurrence, Gauss--Legendre
angles, normalization, derivative multiplication order and flow rules remain
the same. The compact cache stores three polar factors per theta node and
two phase factors per azimuthal node. Each finder logs actual host, device and
unique allocation bytes. This storage count excludes mesh and flow arrays
and is separate from a measured process peak.

The Serial validation drivers accept `--harmonic-storage factorized`; their
allocation screen and recorded commands use that mode, and qualification
requires its actual per-horizon allocation witness. Fixed-order quadrature
inherits the baseline mode. At lmax160, ntheta162 the compact harmonic data
are101,615,592 bytes per horizon including the15 tiny unused dense views,
versus130,814,792,640 bytes for dense tables. These are allocation estimates;
bitwise harmonic, flow-history, horizon and measured-RAM checks are pending
the end of the single-worker benchmark campaign. No physical gate is changed.

`Athena_ENABLE_FASTFLOW_CACHE_TEST=ON` builds `test_fastflow_cache`, comparing
all15 valid components on small production GL grids and selected endpoint,
degree and azimuthal nodes through lmax160 on the actual Kokkos execution
space. `check_fastflow_cache.py` runs separate Serial dense/compact processes
on exact seeds with displaced search centers. Its opt-in
`fastflow/full_precision_trace=true` records every iteration's coefficients,
radii and angular gradients, flow source and integrals at round-trip precision.
The trace requires verbose output and leaves the flow decisions unchanged.
An optional historical dense executable compares ordinary histories and final
outputs against the previous implementation. Equivalence evidence and the
original physical tolerances are required separately; these controls have
been built but have not yet run.
The displaced comparison does not interpret a boosted surface's coordinate
rotation integral as intrinsic spin or impose unit Christodoulou mass on it;
those quantities depend on the rotation origin. Centered exact-seed checks
retain their original mass and spin criteria.

The binary driver has an optional `--enclosure-axis x` (or y/z) range
certificate for its retained finite real harmonic surface. It projects each
degree onto the chosen zonal mode and bounds the orthogonal remainder by the
addition theorem. The zonal Legendre expansion is converted to a finite
cosine series using the [Legendre generating function](https://dlmf.nist.gov/14.7.E19).
Uniform angular samples are supplemented by global first- and second-
derivative bounds, so the returned lower/upper bounds cover the full sphere.
This can tighten an inconclusive monopole-minus-tail bound for elongated
surfaces without assuming exact axial symmetry.

The helper uses 100-digit outward Decimal intervals, exact rational Machin
bounds for pi, integer quadrant reduction and a bounded cosine Taylor tail.
Square roots are explicitly widened because [Decimal sqrt uses half-even
rounding](https://docs.python.org/3/library/decimal.html#decimal.Decimal.sqrt).
Float lower/upper outputs round outward. Coefficient payloads, method source,
axis, subdivision count and all error terms are retained. Worker logs supplying
the actual finder centers and execution witnesses are hash-bound alongside
the shape and summary files. Existing offsets,
modified-ball radii, rounding allowances and the empirical refinement buffer
are still subtracted. The old Cauchy result is also recorded; previous failed
enclosure evidence is preserved. This certifies the ideal finite expansion,
with no continuum PDE/truncation or physical-acceptance transfer. Synthetic
known-extremum and phase/rounding controls are prepared; execution is pending
the end of the benchmark campaign.
