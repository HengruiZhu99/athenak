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

Double precision is required. CMake resolves the header/library only under
HISPID_ROOT, prints those paths, and records the library SHA256. Reconfigure
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
their earlier fingerprints in `hispid-validation.json`. No regular-basis
solved binary or binary attenuation enclosure is accepted.
