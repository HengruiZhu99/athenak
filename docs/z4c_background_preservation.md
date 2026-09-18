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

## Diagnostic working directory

Start each process in its case directory when collecting initialization and
stage traces. AthenaK applies `-d` after problem initialization; using `-d` alone
puts the initial debug CSV rows in the launch directory and subsequent rows in
the run directory. The regression runner already supplies `cwd=run`. GPU job
8836413 exposed this launch-script issue: its first three-step perturbation
control completed, but the strict verifier correctly rejected the incomplete
case directory. The corrected launcher enters the case directory before
starting MPI. The original files and the rejected verification are preserved.

## Inner-layer controls at fixed resolution

At dx=0.25M and 60M, the 0.5/1M freeze/ramp control has max|Theta|=1.20514e-7,
with its maximum at (0.625,-0.125,0.125)M, r=0.64952M, within the sponge. The
xy slice contains the global history maximum. Increasing only the sponge rate
from 5 to 50 reduces the endpoint to 2.20003e-8, but the late log-growth slope
remains positive (approximately 0.0632/M, versus 0.1319/M at rate 5). The source
timestep safeguard is active; Z4c damping remains 0.1/0. This is not a stability
pass and does not justify replacing the resolved vacuum test.

With the original 1/1.4M layer on the same dx=0.25M grid, max|Theta| reaches
7.30406e-7 at 60M. Its xy-slice maximum is 7.07443e-7 at
(1.375,-0.625,0.125)M, r=1.51554M; that slice maximum is not the global argmax.
Changing the freeze radius also changes the analytic interior background
clamp, so this comparison does not isolate a single sponge term. Spatial
profiles, raw exterior Hamiltonian maxima, and changes relative to initial
Hamiltonian error must remain separate diagnostics.

## Actual mesh spacing diagnostic and refined GPU audit

The old `EXCISION_SETUP buffer_cells` divided by a spacing inferred from
`amr_bh_refine_level`, even when the input disabled AMR. The coarse dx=0.5M
control therefore printed 614.4 cells although its actual horizon-to-ramp
buffer was only 1.2 cells. The diagnostic now reports the coarsest actual
spacing among blocks intersecting a conservative horizon bounding sphere,
reduced across MPI ranks; `planned_dx` is labeled separately. Only active
spatial dimensions enter this estimate. The placement defaults and evolution
are unchanged. All sixteen CPU/MPI regressions pass, including actual-spacing
checks (0.5M uniform, 0.25M refined) and exact rank agreement of the physical
perturbation maxima.

Corrected Aurora job 8836424 passed all four small-pulse GPU controls: single
and double dipole amplitudes on one/four MPI ranks. Their physical Theta response
maxima match the CPU reference exactly; odd-parity and algebraic projection
checks pass. Its 1408-block refined vacuum audit at dx=0.0625M, freeze/ramp
0.5/1M, and rate 5 passed on 96 ranks: 1500525 state/RHS rows and 108864 geometry
rows have exactly zero differences and no bit mismatches. This job uses the
pinned 7aefdac3 numerical build; the later actual-spacing diagnostic is a host
reporting change, not an evolution change. The short refined audit does not
establish long-time perturbation stability.

## Per-rank checkpoint continuation

A new restart regression exposed an MPI I/O dispatch error before evolution:
`Mesh::BuildTreeFromRestart` omitted `single_file_per_rank` when calling
`GetPosition`, passing a C `FILE*` to `MPI_File_get_position`. A debugger
backtrace identified this call. Passing the existing flag fixes the abort;
shared-file dispatch is unchanged.

The TOV/KS restart initializer also projected and recast the already evolved
saved residual again. A no-step restart changed active metric components by
4.44e-16. Restart now reconstructs derived full/background fields without that
extra projection/recast. The normal stage projection remains active; no
physical perturbation is clipped or reset. This assumes continuation of a
valid checkpoint with the same background/gauge parameters and MPI partition.
Per-rank files do not support automatic redistribution to a new rank count.

`tst/regression/z4c_background_restart.py` compares six uninterrupted RK3 steps
with three steps plus a checkpoint and three resumed steps, using sixth-order
spatial operators. It also performs a no-step restoration. Uniform and refined
vacuum/dipole cases on one/four MPI ranks pass all eight cases (32 launches):

- Saved active Z4c values are bitwise unchanged by restoration in every case.
- Vacuum residuals, including stored ghosts, remain bitwise positive zero.
- Nonzero perturbations remain finite and nonzero with valid metrics. Maximum
  active differences between continuous and resumed evolution are 3.10e-13
  (uniform) and 1.66e-14 (refined), against residual amplitudes near 5.8e-9.
  Ghosts are reconstructed during restart, so nonzero evolution is not claimed
  to be bitwise identical to uninterrupted evolution. The test bounds this
  difference at 1e-12 for the specified 1e-8 seed.
- All reported response/error maxima agree exactly between one and four ranks.

These tests establish restoration and equilibrium, not long-time perturbation
stability. GPU confirmation is recorded below. Run with, for example:

```sh
python3 tst/regression/z4c_background_restart.py --exe /absolute/path/to/athena \
  --output /absolute/path/to/new-results --launcher mpiexec
```

## Checkpoint metadata initialization

The 384-rank 200M checkpoint from job 8836445 has 1408 blocks and 5158010880
payload bytes, all finite. A strict byte comparison of its headers found
rank-dependent values in exactly nine unused root-mesh coarse-index integers.
All other header bytes match across all ranks. These fields were never
initialized; some contained recognizable text from earlier heap allocations.
The Mesh constructor now value-initializes its region/index structures, and
restart loading canonicalizes only the unused root coarse indices to zero.
Meaningful MeshBlock coarse indices and evolution arrays are preserved. The
checkpoint format is unchanged. This is a metadata initialization bug, not
evidence of a numerical kernel race or an explanation for Theta growth.

The eight-case CPU/MPI restart suite passes with `MallocPreScribble=1`, requiring
zero unused root fields and byte-identical complete headers within each rank
cohort. An additional no-step legacy-file test poisons those nine integers and
checks that continuation emits canonical metadata while preserving exact
vacuum state. All 33 launches have finite histories and valid metrics; the
physical response/error maxima are unchanged from the preceding restart tests.

## Further perturbation controls (not stability passes)

The resolved dx=0.0625M, rate-5 run 8836445 continued to 225M with valid metrics,
but max|Theta| reached 5.7100e-7 from a roughly 1e-8 seed. It was manually stopped
after preserving the 200.00625M checkpoint. This was neither a walltime stop nor
completion of its 1000M target. At 200M, the full 3D maximum is 3.09774e-7 at
(0.46875,-0.09375,-0.15625)M, r=0.502921M, immediately outside the frozen core
(rank 108, block 326, level 4). The exterior Theta maximum is 1.31467e-10 at
r=9.12671M. These are Theta maxima, not Hamiltonian maxima or first-injection
locations.

Two additional dx=0.25M controls reached 60M, finite but still growing. Moving
only the analytic background clamp from 0.5M to 0.125M, while leaving the
residual freeze/ramp at 0.5/1M, changes the 30–60M log-growth slope from
0.132329/M to 0.132198/M. Increasing only residual lapse damping from zero to
two gives 0.129930/M. Neither isolates a sufficient cure. The experimental
clamp option was therefore not retained in production source. Exact equilibrium
with the inward clamp had separately passed uniform/refined one/four-rank
stage audits. These controls constrain hypotheses; they do not identify a
unique erroneous term in the nonzero-residual operator.

## Signed-zero background representation

An axis-aligned control (same spacing, domain translated by half a cell while
the black hole remains at the coordinate origin) exposed signed-zero input
differences at the first reconstruction. `background + 0` can turn `-0` into
`+0`. The run still preserved zero evolution: 870 state-audit rows and 54
geometry-audit rows had bit mismatches but zero numerical difference. This
was not the source of the observed growing perturbation.

Reconstruction now returns the background value directly for an exactly zero
residual, including signed zero, and performs the original addition for every
nonzero residual. There is no epsilon threshold or perturbation reset. The
expanded 24-case CPU/MPI stage suite passes, adding uniform/refined axis-aligned
equilibria and single/double-amplitude axis-aligned pulses on one/four ranks.
All sixteen pre-existing case results, including physical response and
projection error maxima, are unchanged exactly. Axis-aligned zero state/RHS
and geometry inputs now pass the strict bitwise audit; the nonzero response
remains linear and rank independent. GPU validation of this extension remains
pending.

## Excision characteristic placement audit

The old `AllIngoingExcisionRadius` checked only the 1+log lapse speed and
ignored the Gamma-driver shift speeds. For the production Schwarzschild
background (`residual_lapse_f=1`, `shift_Gamma=1`), the outgoing radial light,
lapse, transverse-shift, and longitudinal-shift cones turn outward at
2, 1.089866773, 0.931142464, and 0.708376139M respectively. The automatic
freeze radius is now 95% of the smallest bound, computed from the configured
lapse and shift coefficients. Explicit radii remain unchanged. Unsupported
spin/gauge configurations require explicit radii and report an unavailable
bound, rather than claiming a lapse-only estimate establishes causal excision.
The bound concerns the zero-background radial principal symbol; it does not
prove finite-difference causality or stability of finite perturbations.

Seven executable initialization tests in
`tst/regression/z4c_excision_characteristics.py` pass: default and varied lapse/
shift coefficients, preserved explicit radii, and unsupported spin/advection
with the appropriate explicit-radius requirement. The running 0.5M freeze
already lies inside the corrected 0.708M bound. This bug therefore invalidates
the old default-placement claim, but does not explain the current inner mode.

A separate audit of `scalar_symbol()` in
`analysis/z4c_characteristic/derive_residual_characteristics.py` found a
lapse/longitudinal-shift cone coincidence at r=3.191280621M for the default
gauge. At `G=3, L=4/C`, its characteristic polynomial is
`(lambda-2)^2 (lambda+2)^2 (C*N^2-lambda^2)^2`, while the eigenspace at
lambda=2 has dimension one for generic C,N: the frozen symbol is defective
there. Changing only `residual_lapse_f` to 0.5 removes this particular
coincidence, but the dx=0.25M dipole control still grows through 60M:
max|Theta|=1.46187e-7, 30–60M log-growth slope 0.131123/M (baseline
0.132329/M). This is a separate limitation of the chosen gauge; it has not
been established as the cause of the observed near-excision growing mode.


## GPU restart confirmation and stronger-sponge control

Aurora job 8836582 (one node, pinned 2b965021 executable) passed the eight-case,
33-launch restart suite in 1m32s, exit 0. Uniform/refined vacuum remains exactly
zero on one/four GPU MPI ranks, including restart. All saved active state is
restored bitwise. Metadata initialization and legacy unused-field handling
pass. Within the GPU backend, one/four-rank results match exactly. For nonzero
seeds, resumed versus uninterrupted evolution differs by at most 3.09569e-13
(uniform) and 1.61723e-14 (refined), within the 1e-12 test bound. CPU/GPU nonzero
response maxima are not bitwise identical: the largest difference is about
1.99e-15. Exact equilibrium and bitwise saved-state restoration hold on both.

The resolved rate-50 control 8836528 retained growth: at 200.00625M,
max|Theta|=1.303649e-7 at (-0.84375,0.15625,0.15625)M, r=0.872205M,
rank 302/block 1081/level 4, inside the transition layer. Exterior
max|Theta|=1.316315e-10 at r=9.126712M. The 150–200M global log-growth slope is
0.022390/M (e-fold about 44.7M). The final copied history at 218.00625M has
max|Theta|=2.020576e-7 and valid metrics. The run was manually canceled after
validating its 200M checkpoint (384 files, 1408 blocks, 5158010880 finite
payload bytes). Its exit 271 and 28m25s walltime are not a clean application
walltime stop or completion of the 1000M target. Like the matched rate-5
executable, it predates the restart/metadata fixes; the checkpoint's only
rank-header differences are the nine unused root-mesh coarse indices.

Increasing only the rate-50 ramp endpoint from 1 to 1.5M on the dx=0.25M
control made growth worse: max|Theta|=1.2757e-4 at 60M, log-growth slope
0.256389/M over 30–60M. The run was manually stopped after 60M. A broader
componentwise sponge is therefore not a demonstrated remedy. Investigation
of metric/connection damping compatibility remains experimental; no such
change has been retained in production source. Atmosphere and physical-star
evolution remain gated on resolved perturbation stability.

## Expanded GPU exactness and rejected interior-layer experiments

Aurora job 8836612, using the pinned d165fe06 executable, passed all 24
axis-expanded controls in 2m15s (exit 0). Eight zero-residual cases preserve
state, RHS, and full/background geometry bitwise on one/four GPU MPI ranks,
including refinement interfaces and coordinate axes. Sixteen nonzero-response
cases pass the amplitude and rank checks. CPU/GPU nonzero response maxima
differ by up to approximately 7.33e-16; this is not a claim of bitwise equality
between backends. Algebraic determinant/trace errors remain below 9e-16.

Three default-off local sponge experiments were investigated and rejected as
stability fixes. Their patches, frozen executables, inputs, and raw results
are retained under `review/vacuum-preservation-20260918`; none is enabled or
retained in production source. All used sixth-order volume differences,
dx=0.25M, freeze/ramp radii 0.5/1.5M, rate 50, and unchanged kappa1=0.1,
kappa2=0. The following comparisons distinguish operator checks from long
evolution tests:

* Adding the continuum, linearized metric-sponge gradient term to the Gamma
  source reduced growth relative to the componentwise wide-layer control,
  but did not remove it. With lapse scale 0.5, max|Theta| at 60M is
  2.12924e-7 and the 30–60M logarithmic growth rate is 0.118770/M.
* A nonlinear discrete directional derivative of the contracted connection
  accounts for the metric-sponge contribution without assuming a spatial
  product rule. Thirty-two independent finite-difference derivative tests,
  the 24-case one/four-rank MPI suite, and byte-identical one/four-thread
  stage audits pass. Nevertheless, max|Theta| at 60M is 2.08279e-7 with
  growth rate 0.140228/M. The run was stopped manually at 66.6M. Correcting
  this one coupling is not sufficient to stabilize the system.
* An auxiliary-only layer damps Theta while adding the opposite physical-K
  compensation to Khat, and relaxes the residual Gamma-minus-contracted-
  connection constraint. It leaves the physical metric and K unchanged by
  the instantaneous sponge source. Deep-core zeroing remains active. Its
  24 short MPI cases pass, but a separate perturbed evolution grows. At
  100M, max|Theta|=1.590186e-7 at (-0.375,-0.125,-0.375)M,
  r=0.544862M, rank/block 0, level 0. The gxx residual reaches 2.587428e-6
  at the same radius; Gamma-x reaches 2.065957e-6 at r=0.649519M.
  The 80–100M Theta log-growth rate is 0.087777/M. The run was stopped
  manually at 108M after validating the finite 100M Z4c checkpoint. Small
  early Theta values alone would have given a misleading assessment.

The discrete connection experiment uses separate producer/consumer kernels
and a pack-owned metric-rate scratch view. Inputs remain immutable during
stencil reads; output cells have single ownership. Its local connection
identity concerns the supplied metric-rate stencil. It does not establish
Hamiltonian/momentum preservation, commutation with mesh interpolation, or
stability of the separate frozen-core boundary and RK projections. No extra
synchronization was introduced as a speculative race fix.

A further committed-source vacuum control used lapse scale 0.1 and
shift_Gamma=0.05, freeze/ramp 1/1.5M, rate 5, and an exterior dipole seed at
2.5M. The known lapse/longitudinal-shift coincidence lies inside its frozen
core, and the radial zero-background characteristic bound is 2M. A +/-4M
box was rejected by the CPBC speed-sign guard before the first step; the
accepted +/-8M box retained dx=0.25M around the hole through SMR, with
dx=0.5M outside. At 50.025M on eight CPU MPI ranks, max|Theta|=1.632263e-6
at (-1.375,0.625,0.625)M, r=1.634587M, block 92, level 1. The metric and
Gamma peaks lie at r=1.386317M. Thus this gauge/excision configuration also
fails the perturbation gate. Its queued one-node Aurora duplicate 8836662
was canceled to avoid spending compute on an already rejected candidate.

These are later amplification locations, not first-injection locations.
The established first-injection fixes remain projection idempotence,
consistent contraction rounding, and signed-zero reconstruction. None of
the experiments above establishes a complete explanation or cure for the
remaining growing mode. No atmosphere or star evolution has been launched
under a claim that this vacuum perturbation gate passed.

### Isolating hard freezing from background regularization

A separate vacuum-only diagnostic disables residual RHS/state hard freezing
while leaving the background clamp and radial damping profile unchanged.
It does not disable volume evolution. Both members of each pair start with
2,764,800 bitwise-identical saved Z4c values, including ghosts. Zero-residual
controls with the diagnostic enabled also pass on one/four MPI ranks through
three RK3 steps. This switch is archived as an experiment, not retained in
production or proposed as a replacement for interior zeroing.

All four perturbed controls use eight CPU MPI ranks, dx=0.25M, rate 5,
freeze/ramp profile radii 0.5/1M, the original background-adapted gauge, and
unchanged kappa1=0.1, kappa2=0:

| Background clamp | Hard freeze | Result |
| --- | --- | --- |
| 0.5M | on | At 60M, max(abs(Theta))=1.20565e-7; 30–60M log-growth 0.132478/M. |
| 0.5M | off | At 60M, max(abs(Theta))=9.12920e-8; log-growth 0.132130/M. |
| 0.125M | on | At 60M, max(abs(Theta))=1.06081e-7; continued positive growth. |
| 0.125M | off | Rapid central growth; max(abs(Theta))=5.79983 at 20.025M, r=0.216506M. Manually stopped at 37.2M. |

The smaller-clamp, unfrozen control reaches max|Theta|=7.94372e-5 at 6M
with dt=0.075M. Repeating with dt=0.01875M gives 5.92339e-5 at 6M: temporal
error changes the amplitude, but rapid amplification remains. The 20M
checkpoint is finite and the metric-validity indicator remains clear despite
order-unity residuals. These checks alone therefore cannot define stability.

Hard freezing is not necessary for the slow mode in the regularized
background, and removing it exposes a much faster central mode when the
clamp is moved inward. This comparison rules out a single hard-freeze-only
explanation; it does not identify a complete cause or a stable alternative.

## Nonzero projection audit

`debug_projection_snapshots=true`, together with `debug_balance=true`, now
writes full state and background arrays immediately before/after algebraic
projection. JSON metadata records shape, scalar representation, active-cell
bounds, coordinates, rank, block, logical/root levels, cycle, and RK stage.
Only live blocks are serialized; unused allocation capacity is excluded.
The feature is read-only and adds no synchronization when disabled.

`tst/regression/z4c_projection_snapshots.py` passed four zero-background cases
(one/four MPI ranks, uniform/refined meshes). Snapshot state equals background
bitwise, and projection leaves it bitwise unchanged. A separate nonzero
control produces bitwise-identical saved Z4c state with snapshots on/off.

At cycle 800, t=60M, RK stage 1 of the rate-5, dx=0.25M dipole control,
projection changes the contracted connection derived from the metric by
5.6223273e-9 at (-0.375,0.375,0.375)M, r=0.649519M, rank/block 6,
logical level 1 (root level 1). Evolved Gamma is unchanged. The corresponding
covariant Z change is 5.3384985e-9. Independent 80-digit evaluation at that
point confirms that these changes exceed floating-point cancellation noise.
Over the same audit, the largest changes in reconstructed physical metric
and extrinsic-curvature tensor are 4.22266e-9 and 1.60572e-8 respectively.
The pre-projection determinant error reaches 4.32091e-9; afterwards it is
approximately 1e-15. These are changes to an already nonzero perturbation,
not the first zero-background injection diagnosed earlier.

The current projection rescales g and removes the A trace while keeping chi,
Khat, and Gamma fixed. It therefore changes physical geometry and the
connection constraint when the pre-projection algebraic errors are nonzero.
This is a measured operation to investigate, not proof that projection is
the sole cause of the unstable mode. A representation-preserving candidate
has been checked algebraically in 32 high-precision point tests, but has not
yet been implemented or validated through stencil updates and MPI transfers.

The separate auxiliary-plus-gauge damping trial also failed: max|Theta| at
60M is 8.42809e-8 with 30–60M log-growth 0.103307/M. At 50M the physical
K residual, explicitly reconstructed as Khat+2Theta, reaches 6.58314e-7 at
r=0.649519M. It was stopped manually at 68.8M; its 24 short MPI cases pass,
but its long perturbation gate does not. This experimental sponge remains
archived outside production source. Global kappa1/kappa2 remain 0.1/0.
