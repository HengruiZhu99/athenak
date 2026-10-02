# HiSpID initial-data integration

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
