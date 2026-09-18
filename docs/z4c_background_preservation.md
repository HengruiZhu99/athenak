# Kerr–Schild residual background preservation

The analytic-background residual evolution must preserve the zero solution in
vacuum. This property is separate from stability of nonzero perturbations.
The following audit uses a nonspinning, origin-centered Kerr–Schild background,
sixth-order spatial differences (`nghost=4`), RK3, evolved background-adapted
lapse/shift, characteristic CPBC with `zero_rate`, and the inner excision sponge.

## Identified injections and fixes

1. `UpdateBackgroundState` already projects the background metric and A tensor.
   Initialization and every RK stage then projected the reconstructed full state
   again. Floating-point projection is not idempotent: the second projection
   changed the conformal metric/A by up to 2.220446049250313e-16 even when full and
   background were bitwise identical before projection. This is the first
   nonzero operation in the audited initialization, before any volume RHS.
   Cubic physical-boundary extrapolation magnified the resulting ghost residuals
   (up to 1.91e-10 at outer ghost corners in the initial coarse control).

   `EnforceAlgConstrOn(full)` now leaves an exactly matching finite background
   geometry unchanged. The background itself is still projected normally.
   Representably different full geometries take the original projection path.
   This is an exact equality test, with no magnitude threshold, state clipping,
   compensating subtraction, or disabling of perturbation evolution. It keeps
   the already-projected background as a fixed point while retaining the
   existing nonzero-geometry projection. It does not claim that floating-point
   determinant normalization produces a determinant bitwise equal to one.

2. With projection corrected, the first volume RHS still produced values up to
   4.73e-16 from bitwise-identical full/background inputs. Permitted multiply-add
   contraction can turn `a*b-c*d` into `fma(a,b,-rounded(c*d))`, which need not
   vanish when `a==c` and `b==d`. Disable contraction in the Clang/IntelLLVM
   `CalcRHS` scope and use the source-specific `-ffp-contract=off` option for GCC.
   This gives the two sides identical rounding in the tested builds. No global
   floating-point settings or dissipation/damping parameters are changed.
   Fast-math reassociation and other compiler/backend combinations require their
   own validation; CPU results alone are not a GPU guarantee.

Exact cancellation here means a zero residual state and zero residual RHS for
this finite, stationary vacuum background, including the audited ghost updates
and mesh interfaces. Absolute ADM constraints still contain background spatial
truncation error. Their nonzero equilibrium value is not residual RHS forcing.
Matter sources and physical perturbations are deliberately not subtracted away.

## Reproducing the short regression

Build with `PROBLEM=z4c_tov_ks`, double precision, and MPI enabled, then run:

```sh
python3 tst/regression/z4c_background_balance.py \
  --exe /absolute/path/to/athena --output /new/output/directory --ranks 1 4
```

The runner preserves logs and inputs and rejects an existing output directory.
It tests uniform multi-block vacuum, genuine static refinement interfaces, and
spherical and dipole Theta pulses at two amplitudes. The opt-in
`outer_sponge_test_theta_pulse_dipole_axis` selects x/y/z with 1/2/3;
zero retains the spherical seed. Dipole spatial parity is checked separately. Exact-zero checks cover every audited RK
stage, full/background fields including ghosts, and 126 packed geometry and
first-derivative/auxiliary components. Algebraic projection is checked separately
against determinant/trace errors, and the physical-cell perturbation response
must be nonzero and linear. Ghost extrapolates are checked for finiteness but
are not used as the physical response amplitude.

`z4c/debug_balance=true` enables per-rank CSV diagnostics. The existing
`debug_reduction_stride` controls sampling. Rows include operation, cycle/time,
RK stage, variable, region, rank, global block ID, physical refinement level,
indices and coordinates, maximum difference, and nonzero/bit-mismatch/nonfinite
counts. Radial region boundaries default to 1, 1.4, and 2M and can be set with
`debug_balance_freeze`, `debug_balance_ramp`, and `debug_balance_horizon`.
These radial diagnostics assume the origin-centered spherical test used here.
Regions overlap: the physical-boundary band is one stencil width, and
`refinement_block` marks blocks adjacent to a different refinement level, not
an exact interface distance. `debug_balance_profiles=true` additionally records
all active-cell Theta values after recasting, for spatial-mode analysis.
Diagnostic host mirrors synchronize only when explicitly enabled.

## CPU evidence, 2026-09-18

- Before the fix, one/four OpenMP threads produced identical stage-audit CSVs.
- After both fixes, 151 audited operations over three RK3 steps had exactly zero
  state/RHS differences. All 126 geometry components matched bitwise. One/four
  OpenMP threads agreed.
- Sixth-order vacuum reached 1000M with Theta and other residual history maxima
  exactly zero, finite histories and no bad metrics.
- Four-rank MPI, eight blocks, also reached 1000M with exactly zero residuals.
  The sampled per-stage files contain 407400 state/RHS rows and 21168 geometry
  rows with zero bit mismatches and zero nonfinite counts.
- The short regression passed on one/four MPI ranks with uniform and 120-block
  refined meshes. Post-projection determinant errors were at most 6.67e-16 and
  trace-A errors at most 7.32e-16 in the pulse tests. Doubling the pulse doubled
  the physical-cell response; one/four rank response maxima matched exactly.

## Race/lifetime audit

Full, background, residual, RHS, and RK registers have separate Kokkos backing
allocations; tensor slices alias only the designated backing view. Geometry
kernels read full/background stencils and write the RHS. Reconstruction,
projection, and recasting are pointwise operations. CPBC writes only each owned
cell's Khat/Theta/Gamma/A RHS; its RHS stencil reads are metric/chi/lapse/shift,
which those boundary kernels do not change. Face ownership excludes duplicate
edge/corner writers. Physical-ghost extrapolation reads the fixed active normal
line; directions execute successively on the default execution instance.

`RecvAndUnpackCC` waits for all remote receives via MPI_Test before unpacking;
packing has a device fence before MPI consumes its buffers. Tasks order RHS,
source hook, CPBC, RK update, exchange, physical ghosts, prolongation, and
projection. Background refresh precedes reconstruction and geometry evaluation;
there is no newly introduced cached background lifetime. Kokkos default SYCL
instances share the singleton execution instance and use its queue/event
ordering. No production-wide fences were added. These source checks and CPU
repeatability provide no evidence of a race; GPU repeatability remains a
separate required check.

## Remaining instability — not fixed by exact equilibrium preservation

Radial Theta pulses of 1e-8 and 1e-10 at radius 3M still trigger late growth on
`dx=0.5M`, freeze/ramp=1/1.4M, sponge=5, kappa1=0.1/kappa2=0. Both small-domain
runs become invalid around 270M. Moving the outer boundary from ±4M to ±8M
preserves a growing mode concentrated near r≈1.3M during its linear-growth
interval. Fits over 90–180M give e-fold times approximately 7.55M and 8.35M.
The larger-domain run reaches its 300M time limit with invalid metrics; its zero
process exit is not a scientific pass. This comparison separates the early
injection, later Theta amplification, and late exterior Hamiltonian maxima.
It supports an inner-region amplification mode, without identifying a single
incorrect geometric term or excluding boundary coupling.

Do not infer perturbation stability from the exact-zero vacuum test. Resolved
vacuum perturbation tests, GPU/MPI validation, and then atmosphere and stellar
runs with the same interior zeroing/damping are still required. The production
star job was canceled before starting while this validation is pursued.

## Controlled dipole evidence

The growing shell pattern in the coarse radial-pulse tests is predominantly
odd (dipolar), rather than a spherical constraint offset. Directly seeding an
x-directed dipole with amplitudes 1e-6 and 1e-8 gives a 100-fold early-response
ratio. At 60M their maximum Theta values are 3.13461e-5 and 3.13451e-7.
Halving the smaller-pulse timestep from 0.15M to 0.075M gives 3.13445e-7.
These controls demonstrate an unstable small-signal mode of the coarse spatial
system, rather than a source timestep instability or a zero-state injection.
They do not identify a particular erroneous continuum term.

The twelve-case MPI regression (one/four ranks) passes with the additional
single/double-amplitude dipole seeds. Physical response maxima match across
rank counts; relative odd-parity errors are below 5.8e-9, and determinant/trace
errors remain below 9e-16. These short response checks are not long-time
stability tests. With the deeper freeze/ramp radii 0.5/1M, dipole Theta at 60M
is 1.24155e-5 for dx=0.5M and 1.20514e-7 for dx=0.25M. Both runs remain finite
at that time but exhibit growth; neither is a stability pass. Finer controls
are still required before progressing to matter evolution.

## GPU/MPI evidence

Aurora job 8836373 (one node, Intel PVC/SYCL, MHDTidal/debug-scaling) completed
normally. Uniform stage audits on one/four ranks and a 120-block refinement
audit on twelve ranks preserved exactly zero state/RHS and bitwise-identical
full/background geometry. The four-rank long control reached 1000M in 338.73
seconds of evolution, with 405000 sampled state/RHS rows and 21168 geometry
rows exactly zero. All history values were finite and metric validity checks
passed. This was target completion, not a walltime stop. The pinned executable
contains the same numerical preservation fixes as df5c97f7; its source candidate
predates host-only algebraic diagnostics and documentation/regression additions.
It is not claimed to be a bit-identical build of that commit.

The expanded sixteen-case CPU/MPI regression also injects a dipole centered at
r=4M across genuine refinement interfaces, at two amplitudes and one/four ranks.
These cases preserve a finite, nonzero, linear response with the correct odd
parity, and projected algebraic constraints remain at floating-point precision.
The passive fluid in vacuum controls has stress-energy feedback disabled;
metric exactness does not establish fluid stability or validate matter coupling.
