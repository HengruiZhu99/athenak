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
was checked algebraically in 32 high-precision point tests and subsequently
tested in the single-block control below. MPI transfers were not implemented
for this rejected candidate.

The separate auxiliary-plus-gauge damping trial also failed: max|Theta| at
60M is 8.42809e-8 with 30–60M log-growth 0.103307/M. At 50M the physical
K residual, explicitly reconstructed as Khat+2Theta, reaches 6.58314e-7 at
r=0.649519M. It was stopped manually at 68.8M; its 24 short MPI cases pass,
but its long perturbation gate does not. This experimental sponge remains
archived outside production source. Global kappa1/kappa2 remain 0.1/0.

### Physical-geometry projection control (rejected as a stability fix)

The candidate has now been tested in a guarded single-block vacuum prototype.
For q=det(g)^(-1/3) and T=tr(g^-1 A), it used g'=qg,
chi'=q chi (chi_psi_power=-4), A'=q(A-gT/3), and Khat'=Khat+T.
Theta was unchanged. With contracted discrete metric connections C before and
C' after projection, it updated Gamma by
(C'-C)+((1-q)/q)(Gamma-C), preserving covariant Z_i. Metric stencils read
immutable inputs in separate kernels before/after the pointwise map. Only
Gamma physical ghosts were refreshed through independent residual scratch.
The exact canonical-background branch was retained, without a tolerance.

This was explicitly limited to one uniform vacuum block with outflow BCs;
it is not an MPI/SMR implementation and has been removed from production
source after the failed stability comparison. In particular, preserving Z
can change Gamma inside the frozen core when neighboring metrics are
projected. That ordering would require further excision analysis before any
production use. No matter test used this prototype.

Four short runs (zero/pulse, one/four OpenMP threads) passed. Equilibrium
state/RHS/geometry audits stayed exactly zero and snapshots stayed bitwise
equal to the background. Thread-count comparisons of all audit CSVs and
snapshot bytes were identical. Nonzero response was retained; determinant
and A-trace errors were below 2.7e-15.

The matched long comparison used sixth-order derivatives, dx=0.25M,
32^3 active cells in one block, domain [-4,4]^3 M, RK3 dt=0.075M,
the 1e-8 dipole Theta pulse, freeze/ramp=0.5/1M, rate=5, f=1,
kappa1/kappa2=0.1/0, and the same characteristic outer boundary.
Both runs reached the 60M target with finite state and valid metrics.
Neither is a stability pass:

| Projection | max(abs(Theta)) at 60M | log-growth, 30–60M |
|---|---:|---:|
| Standard | 1.20514255e-7 | 0.13253461/M |
| Physical-geometry candidate | 1.13865714e-7 | 0.13232546/M |

Both Theta maxima are at (-0.625,0.125,-0.125)M, r=0.649519M,
rank/block 0, relative level 0, cycle 800. These are late-state maxima,
not locations of the original exact-background injection. The spatial mode
and growth rate are almost unchanged. The earlier eight-block MPI freeze-on control gave 1.20565e-7 at 60M,
within 0.05% of this single-block standard result; this is not a claim of
bitwise equality across those separate runs.

One-step checkpoint audits at 60M confirm the candidate actually preserved
its intended quantities. Maximum physical-metric, physical-Kij, and covariant-Z
changes were respectively 8.88e-16, 2.66e-15, and 1.21e-14, versus
4.22e-9, 1.60e-8, and 5.33e-9 under standard projection. Thus removing these
projection changes is insufficient to cure the growing mode. This is not a
claim that standard algebraic projection is generally an implementation error.

At the standard control's Theta maximum, cycle 800 stage 1, signed volume
terms are: advection +1.63022e-7, curvature +3.97560e-7, and Z4c damping
-1.19339e-8. The recorded RHS after KO is +5.20973e-7. Evaluating the
configured sponge at that coordinate gives sigma=4.190951/M and a source
approximately -5.05068e-7, leaving +1.59047e-8. These last two values are
inferred from the analytic source and rounded console values, not separate
raw RHS snapshots. The sponge damps Theta at this point; coupled volume
terms overcome it. This local balance does not identify the full unstable
operator or its first injection. A signed spatial growth budget is the next
diagnostic, rather than another production parameter change.

Artifacts are retained locally in `review/vacuum-preservation-20260918`:
`experimental-physical-projection/tracked.patch`,
`projection-trial-short-results.json`, `projection-long-results.json`,
`projection-t60-comparison.json`, `projection-theta-budget-t60.json`, and
`projection-control.png/pdf`. The pinned experimental executable SHA-256 is
`1d5989bf08abc7de8a5563d267b7653f835e4020c29c301771ba5eb0a0db0284`.
The prototype was based on commit 9b817be1. No Aurora job was submitted for
this rejected candidate; atmosphere/star evolution remains behind the
perturbed-vacuum stability gate.

## Signed stage budget and invalid-state handling

`z4c/debug_snapshot_operations` selects comma-separated `DebugBalance` operation
names for raw binary snapshots (no whitespace in the list). It uses the same
`debug_balance` and stride gates as the CSV audit. Metadata now explicitly
records `compare_background`: false denotes a raw residual state or residual
RHS, not a full state from which the supplied background should be subtracted.
Snapshot payloads include ghosts for forensic use; RHS values are defined on
active cells. The filter is a host-only class member, not part of the Options
structure copied into device kernels.

A four-case one/four-rank uniform/SMR regression with six selected operations
passes strict zero-bit checks, alongside the existing projection snapshots.
Saved nonzero Z4c state is bitwise identical with snapshots enabled/disabled.
The added snapshots do not change the evolution operator.

At the standard single-block checkpoint at 60M, the coordinate-volume Theta
L2 logarithmic amplitude budget is:

| Contribution | Rate (1/M) |
|---|---:|
| Curvature/Hamiltonian source | +4.25127108 |
| Advection | -1.32085523 |
| Z4c damping | -0.10797602 |
| KO dissipation | -0.28768900 |
| Excision source | -2.40266307 |
| Outer boundary correction | -0.00012834 |
| Total | +0.13195942 |

The measured RK3 one-step L2 amplitude growth is +0.13195990/M. The excision
source agrees with its configured analytic damping/freeze operator within
1e-21 in absolute RHS units. Theta is bitwise unchanged by projection and
recasting at all three stages. About 76.4% of its squared norm is in the
0.5–1M annulus. This is a signed diagnostic of this late mode, not a physical
positive-definite energy estimate or a proof of its first injection.
Advection can have a positive value at the pointwise maximum while its
whole-domain inner product is negative; these are different diagnostics.

An independent algebraic split assigns +2.626548/M of the curvature source
to Ricci, +1.455922/M to the K-squared term, and +0.168801/M to the A-squared
term. The split uses the serialized full/background tensors; roundoff in
these inferred subterms is distinct from the directly sampled stage RHS.

The standard-gauge comparison could not use the characteristic boundary,
which explicitly supports the adapted gauge only. Matched Sommerfeld controls
with both gauges became invalid near 29M (standard: first sampled bad metric
29.25M; adapted: 29.025M). The adapted run nevertheless returned a normal
60M time-limit exit, while the standard run was manually stopped at 54.225M.
Neither is a stability pass or a usable gauge comparison for the characteristic
boundary. The Sommerfeld zero control stayed numerically zero but emitted
negative-zero RHS bits at the boundary; the strict characteristic-boundary
bitwise tests remain a separate claim.

Fault injection then exposed an independent implementation defect:
`ApplyInnerExcision` replaced nonfinite Z4c residuals and RHS values by zero
outside the frozen core, including exterior cells. A NaN Gamma inserted into
an exterior checkpoint cell was silently erased and the evolution returned
exact vacuum. The fallback is removed. Finite updates retain the same
arithmetic; explicit zeroing remains confined to the prescribed frozen core.
Invalid fluid states outside the inner layer are likewise no longer repaired
by this excision routine (the MHD solver's own policies are unchanged).

With `problem/metric_diag_history=true`, history sampling now checks all full
Z4c fields and ADM extrinsic-curvature components for finiteness. It tests the
leading principal minors of the conformal spatial metric, since positive
determinant alone also admits two negative eigenvalues. By default,
`metric_diag_abort_on_invalid=true` terminates with a nonzero status and reports
time, cycle, rank, block, relative level, coordinates, and local bad-cell count.
MPI failure uses `MPI_Abort`, including when only a non-root rank detects it.
This check occurs at history output times, not every RK stage, and does not
replace stability checks on small but growing finite residuals. An explicit
false setting allows diagnostic inspection; it never restores the erased-NaN
fallback.

`tst/regression/z4c_invalid_state.py` passes finite, valid vacuum plus six
fault cases: NaN Gamma, negative chi, and a finite indefinite metric, each on
one/four MPI ranks. The four-rank faults are reported on rank 3, block 7.
An explicit opt-out reports one bad cell rather than erasing it. The existing
24-case CPU/MPI equilibrium/perturbation suite also passes, with its entire
result dictionary equal to the pre-change signed-zero regression. This fix
prevents invalid evolutions from masquerading as successful completions;
it does not resolve the finite growing vacuum mode. GPU validation follows
separately, and matter evolution remains gated on vacuum stability.

Follow-up CPU checks repeat the invalid-state tests without MPI using one and
four OpenMP threads; all six fail as intended. Rebuilding after moving the
snapshot filter out of device Options and rerunning the snapshot regression
also passes all four MPI cases and the noninterference test.

A nested sixth-order stencil audit, restricted to r<2M so every input remains
inside the captured ghost extent, further splits the Ricci contribution. In
the 0.5–1M annulus, the derivative of the evolved-connection constraint
contributes -3.99238/M and the remaining geometric Ricci part +7.55252/M,
for the previously measured +3.56014/M net Ricci contribution there. The
connection-constraint derivative is therefore not a positive driver in this
particular inner product. Removing it on the basis of its large magnitude
would be unjustified. This is an inferred diagnostic, not a changed equation.

The next vacuum resolution control retains the earlier inward-characteristic
gauge (lapse scale 0.1, shift coefficient 0.05), freeze/ramp 1/1.5M, rate 5,
kappa1/kappa2=0.1/0, exterior dipole seed at 2.5M, and domain [-8,8]^3M.
Increasing the requested SMR level to 2 produces 960 blocks of 8^3 cells:
512 fine blocks with dx=0.125M and 448 outer blocks with dx=0.25M. Tree
balancing also refines the outer grid, so this is a factor-two resolution
comparison, not a change confined to the central region. Eight MPI ranks
complete the three-step equilibrium audit with all 239400 state/RHS rows
and 9072 geometry rows strictly zero. The perturbed CPU pilot was stopped manually after a short throughput
measurement (about eight seconds per step), before any stability conclusion.
A one-node GPU pilot is prepared after validation. The pushed 4d102603 GPU build and its
separate one-node validation are also pending.

### GPU validation of invalid-state handling and snapshots

Aurora job 8836796 completed with exit status 0 in 2m37s on one node in
MHDTidal/debug-scaling. It used code 4d102603 and pinned executable SHA-256
`f327d091fd20cc32889e214b17600fb372fc5ee3b763e587bce3f0db35c686f6`.
All 24 equilibrium/small-response cases passed; the complete result dictionary
is identical to the preceding d165fe06 GPU run (8836612). This covers one/four
MPI ranks, uniform/SMR meshes, coordinate-axis signed zeros, and nonzero seeds.
All six deliberately invalid checkpoints terminated nonzero and reported the
bad state; the four-rank faults were detected on rank 3, block 7. The explicit
diagnostic opt-out retained and reported the bad cell. Selected-stage and
projection snapshots passed strict zero-bit checks on one/four GPU ranks,
and enabling snapshots left the saved nonzero Z4c state bitwise unchanged.

The 960-block refined vacuum pilot is submitted separately as job 8836811:
one node, 12 GPU MPI ranks, one-hour walltime, application guard 00:55:00.
It first runs three exact-zero steps on its own mesh and verifies all stage,
geometry, algebraic, and refinement-transfer audits before starting the
150M dipole evolution. Submission is not a perturbation-stability result.
The small atmosphere and star remain gated on that unresolved requirement.

An independent offline Fourier audit transcribes the 20-field tracefree
principal tangent system and the actual sixth-order first/second/mixed and
biased-advection operators, including eighth-derivative KO dissipation.
Continuum flat-space and anisotropic frozen-KS eigenvalues agree with the
analytic characteristic speeds. Direct plane-wave evaluation of the C++
finite-difference header matches the transcribed symbols within 5.70e-14.
For 495 wave/orientation samples at each of five radii and both gauge choices,
the largest principal real eigenvalue is below 4.05e-14/M; adding the configured
advection/KO gives negative real parts and RK3 amplification below one at
dx=0.125M, dt=0.0375M. This is a sampled eigenvalue check, not an energy estimate
or a proof of stability: it excludes spatial background gradients, curvature
source terms, boundaries, SMR and the sponge, and does not rule out nonnormal
growth or defective characteristic coincidences. No numerical operator was
changed on the basis of this diagnostic.

Artifacts: `gpu-results/8836796/{balance,invalid,snapshot}-results.json`,
`audit-fd-principal.py/json`, and `check-fd-symbol.cpp/json`, under the local
review directory cited above.

### Rejected outward core/layer control

A separate CPU vacuum control moves only the prescribed freeze/ramp radii
from 1/1.5M to 1.5/1.9M at the preceding coarse dx=0.25M, retaining lapse
scale 0.1, shift coefficient 0.05, rate 5 and kappa1/kappa2=0.1/0. Both edges
remain inside the 2M horizon and the zero-background all-ingoing bound. The
analytic-background regularization also follows the freeze radius in this
implementation, so this is not an isolated damping change.

Four-rank three-step equilibrium remains exact: 119700 state/RHS and 4536
geometry audit rows are strictly zero. The eight-rank perturbed run nevertheless
grows and is stopped manually at 50.25M after validating the complete finite
120-block checkpoint at 50.025M, cycle 667. This is neither a walltime stop nor
completion of its 100M target. At the checkpoint, max|Theta|=9.27361e-7 at
(1.625,-0.625,-0.625)M, r=1.84983M, rank 1, block 27, relative level 1, after
the completed RK3 step. Its exterior maximum is 6.16649e-7 at r=2.01168M.
The 30–50M fitted growth rate is +0.187285/M, compared with +0.226226/M for
the 1/1.5M control. The lower global maximum does not establish stability;
the exterior perturbation is actually larger.

The initial exterior Hamiltonian maximum is already 0.0178169 at r=2.04252M:
sixth-order stencils on this coarse grid reach into the clamped background.
Thus zero residual preservation must not be conflated with small raw physical
constraints. This control is rejected, with no production parameter change.
`causal-outer-core-results.json` and `causal-outer-core-comparison.png/pdf`
retain the comparison and spatial profiles in the local review directory.

Job 8836811 subsequently passed its target-mesh 12-rank GPU zero gate:
307800 state/RHS and 13608 geometry rows are exactly zero; determinant and
trace errors are at most 8.89e-16 and 6.67e-16. Initial/final joined checkpoints
made this gate take about ten minutes, with the first roughly 1GB shared file
alone taking about 279 seconds before cycle zero. Before the long process
started, its checkpoint format was changed to `single_file_per_rank=true`,
using the previously validated rank-file restart path. The running zero input
was unchanged. The original long input and submitted manifest were archived;
the effective manifest was verified at installation and again by the zero
verifier immediately before launching the long evolution. The executable and
all physics parameters are unchanged.

At the initial 9M sample, the fine perturbation has max|Theta|=8.08974e-9,
finite history and no bad metric cells. Early timings span roughly 0.51–0.56
seconds per step at dt=0.0375M: approximately 34–37 minutes for 150M or
3.8–4.2 hours for 1000M on this one node, excluding extra setup/output costs.
These are early throughput estimates, not completion or stability claims.
The long process still receives `-t 00:55:00`; because the preceding gate
consumed allocation time, the one-hour PBS deadline must also be monitored.
The initial projected 150M completion fits within it. Neither matter stage
has been launched.

### Refined GPU perturbation fails the stability gate

Job 8836811 was subsequently canceled after its complete 50.025M checkpoint
was validated. PBS reports state F, exit 271, and walltime 29m57s. This was a
deliberate stop for sustained growth, not a clean application walltime stop or
completion of the requested 150M target. The final sampled history is 51.4125M,
with max|Theta|=1.13572e-7 and zero bad metric cells. All 12 rank files at
50.025M (cycle 1334, 960 blocks, 1,043,988,480 payload bytes) have matching
headers and finite stored MHD/Z4c data.

At that completed RK3 step the Theta maximum is 9.91724e-8 at
(-1.0625,0.3125,-0.1875)M, r=1.12326M, rank 4, global block 324, relative
level 2, inside the 1–1.5M sponge. The exterior maximum is 4.54277e-9 at
(-1.9375,-0.3125,0.4375)M, r=2.01071M, rank 6, block 530, relative level 2.
The frozen core remains exactly zero. These are state maxima at a specified
time, not evidence that the initial perturbation was injected at these points.
The imposed dipole was centered at 2.5M. The fitted 30–50M logarithmic growth
rate is +0.0904144/M, versus +0.226226/M for the coarse control. Refinement
slows the instability but does not eliminate it.

Later timing also deteriorated: cycles 1050–1350 average 1.11091 seconds per
step including intervening output, corresponding to 8.23 hours for 1000M at
dt=0.0375M on one node, excluding setup. The earlier 3.8–4.2 hour extrapolation
was not sustained. Neither estimate establishes production throughput.

### Rejected stronger KO and isolated background-clamp controls

Doubling KO dissipation from 0.5 to 1.0 does not stabilize the original
single-block dx=0.25M, freeze/ramp=0.5/1M, lapse/shift=1/1 control. All 24
MPI equilibrium/response tests pass with KO=1, but the fresh dipole has a
30–60M fitted growth rate +0.133190/M, compared with +0.132328/M at KO=0.5.
At 60M its maximum is 1.43921e-7 at (-0.625,0.125,0.125)M, r=0.649519M,
rank/block 0, level 0. Switching the actual baseline growing 60M checkpoint
to KO=1 gives only a brief initial decrease; the 70–100M growth rate is
+0.136163/M and max|Theta|=2.52388e-5 at the finite 100.05M checkpoint.
Both controls were stopped manually. This also demonstrates why a negative
instantaneous signed budget after changing one term is insufficient: the
coupled mode changes under the modified operator.

A separate experimental parameter decoupled analytic-background regularization
from residual freezing, retaining the rejected outward freeze/ramp=1.5/1.9M
control and changing only the background clamp from 1.5M to 0.25M. The default
and override each pass 24 CPU/MPI equilibrium/response cases; the default
result dictionary equals the baseline exactly. The actual target mesh also
passes four-rank strict-zero audits (119700 state/RHS and 4536 geometry rows).
Enumeration of all 133 sixth-order stencil offsets, including mixed
derivatives and biased advection/KO, finds the closest background sample for
an evolving cell at r=0.544862M. Thus no evolving-cell stencil touches the
0.25M clamp. The initial exterior squared-H integral returns from 1.40208e-3
to 1.43011e-6, confirming removal of that particular background artifact.

Nevertheless, at the validated 50.025M checkpoint the peak Theta is
9.22353e-7 at (1.625,-0.625,-0.625)M, r=1.84983M, rank 1, block 27, level 1.
The exterior maximum is 6.16300e-7 at r=2.01168M. The 30–50M growth rate
is +0.187820/M, essentially unchanged from +0.187285/M with the tied clamp.
The experiment was stopped after checkpoint validation; its patch and pinned
binary were archived and the source restored. Neither KO nor clamp changes
were promoted. kappa1/kappa2 remain 0.1/0 throughout these controls.

The exact-zero equilibrium guarantee remains verified, including MPI/SMR/GPU,
but nonzero vacuum stability is unresolved. The atmosphere and physical-star
stages remain unstarted. Artifacts in the local review directory include
`latest-vacuum-controls.json`, `latest-vacuum-controls.png/pdf`,
`gpu-results/8836811/checkpoint50-profile.json`,
`decoupled-clamp-stencil-support.json`, and
`experimental-decoupled-clamp-current.patch`.

### Full volume-operator audit and rejected interior diffusion

An independent NumPy implementation of the vacuum volume operator was checked
against the saved growing standard-gauge mode at 60M, cycle 800, all three RK
stages. It includes the background gradients, nonlinear curvature, lapse
Hessian, extrinsic-curvature contractions, connection evolution, biased
advection, and adapted residual gauge. Its maximum difference from the C++
volume RHS is 4.39e-14; the maximum geometric response is approximately
3.1e-6. A complex-step linearization differs from the actual nonlinear response
by at most 4.92e-13. These checks validate the diagnostic transcription and
the small-perturbation approximation, not the continuum formulation's
stability or the correctness of every boundary operator.

Closed Schwarzschild background values agree with the serialized background
to 1.07e-14 outside the frozen core. Using analytic derivatives and analytic
background advection in the independent volume operator makes the background
geometric RHS smaller than 6.71e-14. This checks the intended stationary
background separately from the discrete residual cancellation guarantee.
Replacing only the background derivative coefficients in the growing mode's
linear response changes the instantaneous Theta logarithmic amplitude rate
from +0.131959/M to +0.127008/M. Other components respond differently: the
global Khat rate drops from +0.131989/M to +0.075364/M while the Gamma-x rate
rises from +0.131985/M to +0.180229/M. No production derivative change is
justified by these instantaneous results; they do not predict the eigenvector
of a modified evolution operator.

For the actual saved RHS, apply the first-order algebraic projection about
the background and consider all 22 evolving fields (fixed B fields omitted).
The coordinate-component L2 Rayleigh rate is +0.131978862/M, with relative
defect ||L u - lambda u||/(|lambda| ||u||)=1.88729e-4. Thus the saved state is
very nearly a separable growing mode of the sampled projected semidiscrete
operator, rather than growth confined to one diagnostic field. This is not a
complete spectrum of the RK/CPBC/SMR evolution. The largest absolute defect
is 6.55e-12 in Azz at (-0.375,-0.125,0.875)M, r=0.960143M, rank/block/level 0,
stage 1; its location is not an injection location.

A separate experimental interior diffusion used the conservative face form
div(nu*h*w*grad(residual)), nu=0.25, with harmonic face weights vanishing
outside the existing 0.5–1M layer. It required a static uniform resolution
within its support (outer SMR permitted), immutable stencil input, prescribed
zero core, and a combined explicit-source timestep bound. The default and
enabled variants each pass all 24 CPU/MPI cases; the default results equal
the baseline exactly. Eight option/timestep rejection checks pass. At all
three stages of the actual growing checkpoint, the C++ source agrees with an
independent stencil calculation within 2.14e-21, is bitwise unchanged outside
the layer, and has nonpositive componentwise diffusion work. One/four-thread
snapshots are bitwise identical. The face-energy identity closes within
5.05e-29. This established implementation properties only.

Both long diffusion controls nevertheless grow. The fresh seed has a fitted
30–60M rate +0.138229/M and max|Theta|=1.92104e-7 at 60M, compared with
1.20514e-7 without diffusion. Its peak moves from r=0.649519M to r=0.892679M,
still inside the layer. Restarting the original growing 60M checkpoint gives
an initial decrease followed by +0.137872/M growth over 70–80M; its validated
80.025M checkpoint has max|Theta|=1.63506e-6 at the same new radius. Both
checkpoints are finite, but neither control passes stability. They were
stopped manually (last histories 76.8M and 94.2M), not on walltime or target
completion. The experimental patch was archived, the source reverted, and
both local build directories rebuilt from the restored source. No GPU or
matter evolution was launched for this experiment.

Local artifacts: `offline-operator-validation.json`,
`full-background-jet-response.json`, `background-derivative-response.json`,
`inner-viscosity-{snapshot,options}-results.json`, `inner-viscosity-results.json`,
`inner-viscosity-control.png/pdf`, and `experimental-inner-viscosity.patch`.
The next diagnostic must treat the coupled operator and its inner closure;
negative work from an isolated added damping term has repeatedly failed to
predict the resulting mode's stability.

## Matched-gauge boundary comparison and core-stencil rejection

A diagnostic core-stencil continuation was tested without changing stored
residuals: constant or linear continuation into the frozen 0.5M core, followed
by algebraic projection of scratch metric/A fields used only by the RHS.
The stored core stayed exactly zero, exterior stencil inputs were byte
identical, and one/four-thread snapshots matched. Both variants nevertheless
regrew after an initial decrease. Restarting the 60M growing checkpoint,
they reached the 120M target with max|Theta|=3.96211e-4 and 5.49878e-4;
70–100M growth rates were +0.134858/M and +0.135767/M. Their complete final
checkpoint payloads were finite. Completion is not stability. The experimental
patch was archived as `experimental-core-stencil.patch` and removed from source.

The characteristic boundary previously rejected `standard_subtract` even
though the volume operator already supported it. CPBC now consistently
selects its gauge coefficients: background lapse/shift and residual lapse
multiplier for `background_adapted`, full lapse/shift and no adapted multiplier
for `standard_subtract`. This selection applies to both the normal modes and
the tangential-principal datum. The geometric coefficients always use the
full state. The existing validity guards remain in place. This adds a matched
boundary comparison; it does not change the production gauge or cure growth.

Validation of the new path:

- The characteristic numerical algebra checks pass all 103 coefficient cases,
  including finite standard-gauge lapse changes; maximum error 7.51e-14.
- `z4c_background_balance.py --gauge standard_subtract` passes 24 one/four-rank
  CPU cases with exact vacuum stage/RHS/geometry cancellation, uniform and
  refined meshes, coordinate-axis signed zeros, and nonzero linear response.
  Repeating with `--boundary-source tangential_principal` passes another 24.
- `z4c_standard_cpbc.py` seeds a finite 0.01-amplitude lapse/shift pulse at the
  boundary. Independent full-state scalar characteristic rows have residual
  at most 1.11e-17 over nine RK stages. Substituting a background-only lapse
  coefficient instead produces 8.80e-6 error. Actual boundary updates are
  byte identical on one/four ranks and when the adapted-only lapse multiplier
  changes from 1 to 0.37. No small-residual reset is involved.
- On the original growing 60M checkpoint, an independent nonlinear volume
  operator agrees with the standard-gauge C++ RHS to 4.39e-14 over all three
  stages. Thirty one/four-thread input/RHS snapshots match byte for byte.
  Twenty-four unchanged adapted-gauge snapshots match the earlier executable
  byte for byte, including background inputs, volume RHS, and KO.
- This extension has been built and tested on CPU/MPI/OpenMP. Its enabled
  standard-gauge path has not yet been validated on SYCL GPUs. The selection
  reads immutable full/background views; it adds no stencil writes, view
  ownership changes, cache invalidation, task dependencies, or fences.

The matched long controls reject the adapted gauge as a sufficient explanation
of the observed instability. Standard subtraction retains +0.132582/M growth
from 30–60M, versus +0.132329/M for the adapted baseline. At 60M its Theta
maximum is 7.24181e-8 at (-0.625,0.125,0.125)M, r=0.649519M, inside the
0.5–1M sponge. The exterior maximum is 3.30384e-9 at
(-1.875,0.625,0.375)M, r=2.011685M. These are completed-step state maxima,
rank/block/relative-level zero, not first-injection locations. Normalized
Theta profiles have correlation 0.999358; all 22 evolving fields together
have correlation 0.991410. Thus changing gauge alters amplitude more than the
mode shape or growth rate.

The standard-gauge restart control has +0.131400/M growth over 70–100M.
Its validated 100.05M checkpoint has max|Theta|=2.30182e-5 at
(-0.625,0.125,-0.125)M, r=0.649519M. Both fresh and restarted checkpoint
payloads are finite with valid sampled metrics. The runs were stopped after
these checkpoints, with final histories at 65.025M and 115.2M, respectively;
neither reached its requested 120M target or a walltime limit. Atmosphere and
star gates remain closed.

Local artifacts include `core-stencil-control-results.json`,
`standard-cpbc-{balance-mpi,tangential-balance-mpi,finite-mpi-bitwise}/results.json`,
`standard-cpbc-stage-results.json`, `standard-cpbc-{fresh,restart}-omp4/control-results.json`,
and `standard-cpbc-control.png/pdf` with `standard-cpbc-mode-comparison.json`.

The geometric Khat, Theta and conformal-connection RHS terms were also checked
against equations (3)–(6) of [Hilditch et al., arXiv:1212.2901](https://arxiv.org/pdf/1212.2901).
In particular, the connection uses the contracted metric Christoffel symbol
in the shift-gradient terms and the combination 2*dKhat+dTheta in its
constraint coupling. These terms match the reference formulation; this
comparison has not identified a missing sign or factor that explains the
mode. It does not establish stability on this curved background with the
implemented excision boundary. Results for other formulations, such as
[Garcia-Saenz et al., arXiv:2501.01055](https://arxiv.org/html/2501.01055v2),
concern different puncture and constraint-damping systems and do not justify
changing the requested kappa1=0.1, kappa2=0 here without a separate diagnosis.

The new standard-gauge path also passes outgoing lapse, longitudinal-shift,
and transverse-shift pulse tests at dx=0.0625M with fourth-order ghost
extrapolation: the measured interior incoming/outgoing L2 ratios are
0.005110, 0.001560, and 0.005114 (all below the unchanged 0.02 limit).
These tests run to 12M and measure the saved interior state, separately from
boundary enforcement diagnostics. At dx=0.125M the lapse ratio is 0.021018
with fourth-order extrapolation, failing the same limit. With second-order
extrapolation it is 0.028304 for both the new standard gauge and an unchanged
adapted-gauge comparison. These coarse failures remain recorded; no tolerance
was relaxed. This checks three gauge families at normal incidence, not a new
all-orientation reflection validation or a long-time black-hole stability pass.

## Analytic derivatives and covariant constraint-term controls

Two further vacuum-only prototypes were tested against the same sixth-order,
32-cubed single-block control (dx=0.25M, box +/-4M, freeze/ramp=0.5/1M,
rate=5/M, kappa1=0.1, kappa2=0, adapted gauge, KO=0.5). Neither passes the
finite-perturbation stability gate. Both patches and their executables remain
archived locally; neither experimental change is retained in production code.

The first replaced finite-difference background derivatives with analytic
Schwarzschild derivatives, while retaining the finite-difference residual
response. The full and background evaluations use the same modified jets.
It passed one/four-thread exact-zero stage audits (22,650 state/RHS and 1,134
geometry records per run), 24 one/four-rank uniform/refined/axis balance and
nonzero-response cases, independent nonlinear RHS comparisons (4.39e-14
maximum error), and 30 byte-identical thread-count snapshots. Default-disabled
snapshots were also byte identical to the baseline. Nevertheless, fresh and
restarted controls retained growth rates +0.132023/M (30–60M) and +0.131617/M
(70–100M). Complete 60M and 100.05M checkpoint payloads were finite. These
runs were manually stopped at 64.2M and 107.25M, respectively, rather than
reaching their requested 120M target. Analytic background jets do not cure the
observed mode.

The second added nondamping covariant Z4 constraint terms, evaluated for full
and background states and subtracted identically. The existing Z4c damping
normalization was deliberately preserved: this is a formulation comparison,
not a full CCZ4 implementation or an identified transcription fix. In
particular, no change to kappa1 or kappa2 was made. An independent tensor
identity check over 100 positive-definite geometries agreed to 1.43e-14;
all added terms vanished exactly when Theta and the connection constraint
were zero. One/four-thread zero audits and 24 MPI uniform/refined/axis and
physical-response cases passed. Thirty enabled thread snapshots and thirty
default-disabled baseline snapshots were byte identical. The independent
nonlinear RHS comparison agreed to 4.39e-14; the added source was as large as
1.10e-6, so the test did exercise the new terms. Neither enabled prototype was
tested on GPU.

Both constraint-term controls reached the 120M target, with all checkpoint
payload values finite and no sampled invalid metric. Their completion is
not stability:

| Control | max abs(Theta) at 120M | Growth rate, 100–120M | Maximum position |
| --- | ---: | ---: | --- |
| Fresh 1e-8 dipole seed | 2.1402184e-7 | +0.0488109/M | (-0.625,0.125,0.125)M |
| Restart original growing state at 60M | 1.6955390e-6 | +0.0531386/M | (0.625,-0.125,0.125)M |

Both maxima lie at r=0.649519M in the sponge, at completed RK steps,
rank/block/relative-level zero. Frozen cells remain exactly zero. Exterior
maxima are 9.07911e-9 and 7.27303e-8, respectively, both at r=2.011685M.
These state peaks do not identify the first injection or establish that the
sponge itself supplies positive growth. The instantaneous original-mode Theta
rate had become negative under the added terms, yet a growing mode remained
in the evolution. This is further evidence that an isolated source budget is
not a substitute for analysis of the coupled operator.

The original growing mode was also compared with an infinitesimal translation
of the analytic black hole. Its lapse correlation is high (-0.962), but the
physical metric and extrinsic-curvature correlations are only -0.721 and
-0.465 over evolving cells, with inconsistent best-fit displacements. This
does not support classifying the mode as a pure black-hole translation.

Artifacts: `experimental-analytic-background-jets.patch`,
`experimental-constraint-completion.patch`, the corresponding
`*-stage-results.json` and `*-balance-mpi/results.json`,
`constraint-completion-{fresh,restart}-long-omp4/control-results.json`,
`constraint-completion-instantaneous.json`, and
`mode-translation-comparison.json`. Atmosphere and star gates remain closed.

### Coupled linearization and stage-order checks

An independent diagnostic now assembles the linear response of all 22 evolving
fields about the saved projected background. Cell-local coefficients are
complex-step derivatives of the independently checked nonlinear equations;
finite-difference, advection, KO, ghost extrapolation, inner-layer, boundary,
and projection maps are represented explicitly. This is a diagnostic for the
single-block control, not new production evolution code or an MPI/SMR spectrum
validation.

At the growing 60M checkpoint, its volume, post-KO, post-excision, and
post-boundary predictions agree with saved C++ data through all three RK
stages: maximum absolute RHS difference 4.92e-13, versus a volume response
of about 3.1e-6. The projected semidiscrete operator recovers the measured
+0.131979/M mode rate, with relative eigenvector defect about 1.84e-4.
Its independently implemented transpose satisfies the random-vector dot-product
identity to 4.45e-15 relative error.

Stage ordering matters. Extrapolating already projected active data is not the
same operation as extrapolating the raw RK update and then projecting every
cell, including ghosts. The diagnostic therefore also implements the latter,
actual RK3 sequence. Direct extrapolation of saved raw RK states agrees with
C++ ghost values to 8.59e-19 absolute error. Applying the linearized projection
to saved post-boundary states agrees with recast residuals to 1.31e-13;
this comparison is against a finite, nonlinear perturbation. Predictions of
the next two RK active states agree to 1.15e-13 absolute error, about 1.06e-7
relative. The complete RK map and its transpose satisfy the dot-product
identity to 2.49e-16 relative error.

The complete ghost-array comparison is less accurate (up to 2.49e-9 at outer
corners) when propagating linearization errors through the full stage. The
isolated extrapolation check above still passes. This larger corner error is
recorded separately; it is neither a zero-background preservation failure nor
evidence that the growing physical-domain mode originates at the boundary.
A preliminary eigensolver output failed unit-norm/eigenpair-residual checks
and was rejected. No new evolution change is justified from unvalidated
spectral output. Artifacts include `mode-linearization-validation.json`,
`mode-rk-validation.json`, `mode-stage-transfers.json`, and the locally retained
`mode-linear-operator.py` with its coefficient builder and verifiers.

A subsequent targeted solve of the complete linear RK3 map converged to a
real amplification factor 1.0099483347103 per 0.075M step: growth
+0.1319890106/M, e-folding time 7.5763883M. Its independently recomputed
relative eigenpair residual is 4.46e-10. A separate ARPACK solve in equivalent
90,112-dimensional parity coordinates gives +0.1319890270/M, unit eigenvector
norm, and 7.94e-9 residual. The symmetry reduction preserves all tensor/vector
reflection signs; its transpose check has relative error 6.34e-15.

The converged mode's Theta shape correlates with the saved 60M nonlinear run
at 0.999999989 over evolving cells and 0.999999926 in the exterior. Its
largest normalized Theta lies at r=0.649519M, with the exterior maximum at
r=2.011685M. Thus the measured growth is reproduced by a converged mode of
the discrete update, not merely an instantaneous positive source budget.
This establishes an unstable mode in the seeded x-dipole symmetry sector on
the single-block mesh; it does not establish the full spectrum, its causal
source term, or refined-grid stability. An adjoint remains necessary before
using biorthogonal source sensitivities to select another change. Two
random-field checks also agree with the independent complex-step volume
response to 2.95e-16 relative error. See `right-mode-analysis.json`,
`mode-linearization/right-{targeted-results,octant-eigenvectors}.json`, and
`mode-random-response.json`.

The adjoint subsequently converged as well. A separately checked left RK
mode has residual 4.23e-11 against the right-mode eigenvalue. In the specified
coordinate-component Euclidean norm, the left/right eigenvector condition
number is about 480.12. This quantifies significant nonnormality; it is not
a physical energy norm. Differentiating the actual three-stage map, including
projection and ghost ordering, gives the following growth-rate derivatives
with respect to hypothetical multiplicative changes of individual operator
terms:

| Term multiplier | d(growth rate)/d(multiplier), in 1/M |
| --- | ---: |
| Advection | -1.357893 |
| Algebraic response to A | +0.969140 |
| First derivatives of the conformal metric | +0.389294 |
| KO dissipation | +0.016099 |
| Inner sponge | +0.030963 |
| Characteristic boundary correction | -0.010409 |

These are coupled eigenvalue sensitivities, not a sum of physical energies
and not permission to rescale Einstein-equation terms. The largest local
advection and algebraic-A sensitivities occur at r=1.709349M; the strongest
metric-first-derivative sensitivity is at r=1.815730M. These lie beyond the
0.5–1M sponge but inside the horizon, while the Theta state maximum remains
at r=0.649519M. Thus a state maximum is not a reliable locator of the operator
feedback most influential on this mode.

An independent centered finite change of the entire RHS multiplier agrees
with the differentiated RK sensitivity to 2.1e-13 in the growth-rate derivative.
Increasing only the sponge rate from 5 to 5.05 in the linear operator gives
growth +0.1322964028/M, with a unit-norm eigenvector and independently checked
residual 6.81e-11. The measured increase, +0.0003073922/M, agrees with the
local sensitivity prediction +0.00030963/M to within 0.8%; finite changes need
not equal their first derivative exactly. This is a linear-operator control,
not a new C++ long evolution. It confirms that stronger local negative
relaxation need not decrease a coupled mode's growth rate. The first ARPACK
output for the 4.95-rate control failed norm/residual validation and was
rejected; its reported eigenvalue is not evidence about the evolution.

The validated adjoint also predicts a negative initial growth-rate derivative
for the previously rejected nondamping constraint-completion direction
(-0.136875/M). The finite completion experiment nevertheless retained a
+0.05/M mode. This provides a concrete reason to validate the changed spectrum
and nonlinear evolution rather than extrapolate a local derivative to a full
formulation change. No arbitrary rescaling of physical A/metric terms, small
residual reset, or global suppression of evolution has been introduced.
Atmosphere and star runs remain gated on perturbation stability.

Artifacts: `mode-sensitivity.json`, `mode-sensitivity.png/pdf`,
`mode-linearization/left-targeted-large-results.json`,
`mode-linearization/sponge-1.01-eigenvectors.json`, and
`constraint-completion-eigen-sensitivity.json`. The independent volume,
RK, transpose, eigenmode, and source-sensitivity tools are retained under the
local review directory. These diagnostics have not established a stable
replacement discretization or a new production fix.

The requested vacuum-to-MPI gate was repeated on the restored production
numerics (source HEAD `5b089281`, MPI executable SHA256
`2a6d1c7eadfc5db2a04258ae3a234aa78a39021d56402b5f8101508d4a9fa44c`). All
24 cases passed on one/four MPI ranks with one OpenMP thread: eight exact
equilibria and sixteen nonzero-response cases, including sixth-order SMR
interfaces and coordinate axes. Across the exact cases, 525,750 sampled
state/RHS rows and 22,680 auxiliary-geometry rows were bitwise zero. This is
a repeat of the equilibrium/short-response gate, not a new long-time
perturbation-stability pass. Results are in
`sequence-vacuum-mpi-recheck/results.json` and its summary JSON.

The rate-4.95 eigenpair subsequently passed independent validation: growth
+0.1316771145/M, residual 5.56e-10. Together with rate 5.05, the centered
sponge-multiplier derivative is +0.0309644117/M, within 3.59e-5 relative of
the adjoint derivative. Both rates remain unstable. See
`sponge-eigen-sensitivity-validation.json`.

Additional constraint-completion directions were checked with cached
complex-step Jacobians. Their response agrees with the archived C++
completion prototype to 5.24e-13 absolute error across three RK stages;
the transpose error is 8.13e-16. On 300 finite, positive-definite,
algebraically constrained states with Theta=0 and Gamma constraint=0, every
added term vanishes exactly while the physical geometric RHS remains
nonzero. Nonetheless, validated candidate modes still grow:

| Diagnostic variant | Growth (1/M) | Eigenpair residual |
| --- | ---: | ---: |
| Completion, kappa3=0.5, original Z4c damping | +0.05532445 | 7.63e-9 |
| Same, half the adapted lapse multiplier | +0.04875351 | 4.63e-9 |
| Reference CCZ4 Gamma damping normalization, kappa3=0.5 | +0.05229492 | 9.10e-8 |
| Background-scaled shift gauge, original Z4c equations | +0.12662872 | 2.76e-10 |

The CCZ4 reference changes the Gamma damping operator even though the
numeric kappa1 remains 0.1; it is not an unchanged-damping fix. The separate
kappa3=1 targeted solve missed its declared residual tolerance and is not
used as a validated eigenpair. The already recorded nonlinear C++ control
independently rejects that candidate. No library convergence flag replaces
the explicit eigenvector norm and residual checks.

A further diagnostic removed all active-stencil reads of a small cubical
frozen core. It retains sixth order in the bulk, uses centered fourth/second
order where necessary, and second-order outward derivatives at 48 cells.
Every outward-stencil point has inward characteristic speeds, with minimum
margin 0.1393. Polynomial, independent volume-response, and RK transpose
checks pass at 1.92e-13 absolute, 3.20e-16 relative, and 7.18e-16 relative,
respectively. This construction is not claimed to be an SBP stability proof.

Despite those checks, a converged complex mode has growth +0.12118839/M,
angular frequency 0.11251736/M, unit norm, and eigenpair residual 1.64e-10.
Its Theta maximum is at (-0.375,-0.375,-0.125)M, r=0.544862M; its exterior
maximum is at (-1.875,-0.625,-0.375)M, r=2.011685M. These are single-block,
level-zero diagnostic eigenmode locations, not first-injection coordinates
from a nonlinear C++ run. Avoiding frozen-cell reads alone is therefore
insufficient. This candidate was not promoted to production. See
`core-avoiding-operator-validation.json` and `core-avoiding-mode-analysis.json`.

A coordinate diagnostic also uses the R0=M analytic Schwarzschild trumpet
of [Dennison and Baumgarte](https://arxiv.org/abs/1403.5484), equations 15–20.
This is not the stationary 1+log trumpet. Its continuum geometric RHS and
Hamiltonian checks are below 3.4e-15 on 200 points; the independent linear
volume-response check is 2.55e-16 relative. With a prescribed, background
scaled gauge, sixth-order dx=0.125M, and unchanged kappa1=0.1/kappa2=0, a
compact perturbation initially decays but later grows: Theta's fitted
31.5–45M slope is +0.09608/M. The horizon is at coordinate r=M, areal R=2M;
these radii must not be conflated with Kerr–Schild coordinates. This is
another rejected linear diagnostic, not a nonlinear or MPI evolution pass.
An independently rechecked unit-norm eigenvector subsequently confirms
+0.09452981/M growth for the adapted-gauge trumpet map, with residual
7.75e-10. Its Theta maximum is at coordinate r=0.324760M (areal
R=1.324760M); the exterior maximum is near the small box's outer face,
at coordinate r=1.939515M (areal R=2.939515M). The Gamma constraint and
Theta-RHS Hamiltonian-like quantity are both nonzero in this mode. See
`trumpet-mode-analysis.json`; this eigenmode is distinct from the finite-time
full-gauge snapshot below.

Restoring the full lapse/shift-advection response on that trumpet does not
resolve the issue: a separate 45M compact-pulse test has late fitted growth
+0.09007/M in Theta, +0.08310/M in the Gamma constraint, and +0.08722/M in
the Theta-RHS Hamiltonian-like diagnostic. At 45M, the Gamma-constraint maximum is at
(-0.0625,0.0625,-0.3125)M, coordinate r=0.324760M, areal R=1.324760M.
Theta and that Hamiltonian-like diagnostic peak at (0.3125,0.0625,-0.1875)M,
coordinate r=0.369755M, areal R=1.369755M. These lie in that diagnostic's
inner damping annulus. They are later amplification locations, not evidence
of first injection. The independent complex-step constraint-profile check
agrees to 2.52e-16 relative. See `trumpet-standard-transient-results.json`
and `trumpet-constraint-profile-validation.json`.

The full-gauge diagnostic initially failed its independent response check
because its changing-shift coefficient used a centered background gradient;
the implemented advection uses a biased gradient. Correcting that diagnostic
coefficient gives 2.55e-16 relative agreement. The failed version was never
used for an evolution or promoted to production.

The tracked production equations remain unchanged by these candidate tests.
The atmosphere and physical-star gates remain closed until a vacuum
configuration passes finite-perturbation stability as well as exactness.

The follow-up inner-sponge comparison replaces relaxation of all residual
components by relaxation of Theta and the Gamma constraint only, retaining
the deep frozen core and the original Z4c kappa terms. Its independent source
and RK transpose checks pass at 2.73e-15 and 1.37e-15 relative; zero remains
exact in the linear map. Nevertheless, its 45M compact-pulse control has
31.5–45M growth +0.08695/M in Theta, +0.08040/M in the Gamma constraint,
and +0.08181/M in the Theta-RHS Hamiltonian-like diagnostic. The latter two peak at coordinate
r=0.207289M (areal R=1.207289M); Theta peaks at coordinate r=0.569402M
(areal R=1.569402M). This comparison also fails the stability gate. It does
not establish that relaxing the metric components is the sole cause.
See `trumpet-constraint-sponge-results.json`; all these are local linear
diagnostics, and no new nonlinear production candidate was submitted.


## Independent ADM diagnostic and further vacuum controls

The earlier offline trumpet JSON keys `delta_Hamiltonian` were misleading:
they reconstruct the Hamiltonian-like quantity in the Theta RHS, whose Ricci
expression contains derivatives of the independently evolved conformal
connection. When its constraint is nonzero, that is not the physical ADM
Hamiltonian. This changes diagnostic interpretation, not the existing C++
`ADMConstraints` implementation or the observed Theta growth. Earlier raw
JSON files remain available, with this correction applying to their labels.

`analysis/z4c_characteristic/adm_hamiltonian.py` now independently evaluates
`R(gamma) + K^2 - K_ij K^ij` directly from the physical spatial metric, its
first/second derivatives, and extrinsic curvature. It neither projects data
nor subtracts a background, and does not read evolved Gamma. The local grid
wrapper differentiates `gamma_ij = gtilde_ij / chi`; it reports the directional
residual of the raw finite-difference ADM diagnostic. Raw background H and
residual H remain distinct.

A Gamma-only perturbation provides a concrete counterexample: the physical
ADM Hamiltonian residual is exactly zero, whereas the old Theta-derived
quantity has maximum 0.16201857 in the same linear diagnostic normalization.
For the validated adapted-gauge trumpet eigenmode, the independent ADM
Hamiltonian residual is nonzero. Its maximum is at
(-0.4375,-0.0625,-0.0625)M, coordinate r=0.446339M, areal R=1.446339M;
the exterior maximum is at (-0.4375,1.9375,-0.0625)M, coordinate r=1.987264M,
areal R=2.987264M. These are normalized eigenmode diagnostics on rank/block/
relative-level zero, not first-injection locations or a nonlinear evolution
checkpoint. See `hamiltonian-diagnostic-distinction.json`.

The committed `check_adm_hamiltonian_numeric.py` checks 200 points per case:
a conformally flat curved metric, a nonorthogonal affine coordinate change,
a flat metric in nonlinear coordinates, analytic Kerr-Schild Schwarzschild
vacuum data spanning r=0.4–10M, and complex directional derivatives. Maximum
analytic errors are 8.9e-16 for the first coordinate tests and 2.14e-14 for
Schwarzschild; the complex versus centered derivative difference is
2.68e-11. Inputs remain unchanged. Run it with a NumPy-enabled Python:

```sh
python analysis/z4c_characteristic/check_adm_hamiltonian_numeric.py
```

A sixth-order product-rule correction to variable-shift advection was also
tested offline: `[D(beta*u) - beta*D(u) - u*D(beta)] / 2`, retaining the
original biased advection and KO. Its nonlinear-response and RK transpose
checks give 1.01e-15 and 7.81e-15 relative error. A periodic scalar check
converges at orders 5.92 and 5.98 and satisfies the corresponding discrete
energy identity within 1.67e-15. Those properties do not ensure stability of
the full tensor/boundary/RK problem: an independently validated unit-norm
mode still grows at +0.13685359/M, residual 2.83e-10. Theta peaks at
(-0.625,-0.125,-0.125)M, r=0.649519M; its exterior maximum is at
(-1.875,-0.375,-0.625)M, r=2.011685M. This consistent discretization candidate
was rejected, not promoted to C++. See `split-advection-mode-analysis.json`.

The four saved KS, split-advection, core-avoiding KS, and adapted-trumpet
modes were independently rechecked with frozen coordinates explicitly
removed. They already had exactly zero core values; the subsequent RK step
also leaves the core exactly zero. All eigenpair residuals remain below
8e-10. An initial concern about random seeds in the frozen core was disproved:
`raw_active` calls `ghost`, which masks that core before the first step.
Explicit pre-zeroing reproduces the earlier numeric records exactly, apart
from elapsed wall time. Additional assertions now check the core after every
step. This was a verification improvement, not an identified initialization
bug. See `frozen-core-eigenmode-revalidation.json` and
`transient-core-initialization-correction.json`.


The completed controls with explicit core-zero assertions remain unstable:

| Local linear diagnostic | Target reached | Late fit interval | Theta growth (1/M) |
| --- | ---: | ---: | ---: |
| Trumpet: constraint layer to r=0.5M | 60M | 45–60M | +0.07166064 |
| Trumpet: constraint layer to r=0.9M | 60M | 45–60M | +0.07199307 |
| Kerr–Schild: connection adjustment | 90M | 60–90M | +0.07458832 |

Both trumpet controls keep the coordinate horizon at r=M, the frozen core
at r=0.125M, and the maximum sponge rate at 5/M. Only Theta and the Gamma
constraint are relaxed; extending the outer edge from coordinate r=0.5M to
0.9M does not remove late growth. The fitted Theta increases by factors
2.93 and 2.95, respectively, over 45–60M. Both retain zero outer-boundary
initial perturbations and exact zero residuals in the core at every step.

The Kerr-Schild diagnostic adds
`-C^j partial_j beta^i - (2/3) C^i partial_j beta^j` to evolved Gamma, with
`C^i = Gamma^i - Gamma_metric^i`. This is motivated by equation (45) of
[Yo, Baumgarte and Shapiro](https://arxiv.org/abs/gr-qc/0209066), relative to
the current use of contracted metric Gamma in the shift-gradient terms.
It is not the published BSSN system. Original kappa1=0.1/kappa2=0 terms
remain unchanged, but this adds constraint couplings and therefore changes
the total constraint operator. Independent nonlinear-response and RK
transpose errors are 4.47e-16 and 5.25e-15; the addition vanishes exactly on
328 finite C=0 states while the physical RHS remains nonzero. Its local
adjoint growth derivative is -0.12940471/M, confirmed by a centered change
of the full RK map. Nevertheless, the complete-strength 90M control still
has +0.07458832/M late growth, increasing Theta by a factor 9.36 over
60–90M. A favorable local sensitivity did not provide a stable update.

At the final completed steps, the independent physical ADM Hamiltonian
residual peaks at (-0.1875,-0.0625,0.0625)M in both trumpet controls
(coordinate r=0.207289M, areal R=1.207289M). In the connection-adjustment
Kerr-Schild control it peaks at (0.125,-0.125,0.625)M, r=0.649519M, whereas
Theta peaks at (0.375,-0.125,0.375)M, r=0.544862M. These single-block,
level-zero local snapshots identify amplification locations, not first
injection. Complete maxima, exterior maxima, fit intervals and normalization
bookkeeping are in `latest-vacuum-control-results.json`; the reviewed plot
is `vacuum-latest-controls.png/pdf`.

All three runs ended normally at their prescribed diagnostic targets, not
at walltime. None is a nonlinear, refined-mesh or MPI stability pass. No
experimental equation change was promoted to production, and no atmosphere,
physical-star or new Aurora production run was started. The previously
verified vacuum/MPI exactness remains distinct from the unresolved finite-
perturbation stability gate.

### Physical momentum diagnostic and further inner-layer controls

`analysis/z4c_characteristic/adm_momentum.py` evaluates the vacuum momentum
covector `D_j K^j_i - D_i K` from the physical metric, covariant extrinsic
curvature, and their first derivatives. It does not use evolved conformal
Gamma. Like the independent Hamiltonian helper, it performs neither
projection nor background subtraction. A raw ADM constraint and the
full-minus-background constraint must be labeled separately.

Run its independent checks with:

```sh
python analysis/z4c_characteristic/check_adm_momentum_numeric.py
```

The regression uses 200 samples per case. It checks `K_ij=k(x) gamma_ij`
against the analytic momentum `-2 partial_i k`, a nonorthogonal affine
coordinate transformation of the covector, analytic Schwarzschild data
including the horizon interior, complex derivative-only inputs, and an
independent conservative-coordinate identity on arbitrary symmetric metric
and extrinsic-curvature jets. Maximum errors are 3.34e-16, 3.34e-16,
2.14e-14, 2.23e-16, and 1.56e-15 respectively. Inputs remain unchanged.

Three additional local linear controls were rejected:

| Diagnostic | Target/estimator | Theta growth (1/M) |
| --- | --- | ---: |
| Half constraint completion plus connection adjustment | 60–90M fit; 90M reached normally | +0.05191454 |
| Single-component determinant/trace projection | 60–90M fit; 90M reached normally | +0.12761410 |
| Inner physical-Hamiltonian relaxation | Validated RK eigenpair, dt=0.05M | +0.13187844 |

The first two amplify Theta by factors 4.76 and 45.99 over 60–90M.
Their final Theta maxima are at (-0.125,0.125,1.125)M and
(-0.125,0.125,0.625)M, respectively, on rank/block/relative-level zero after
a completed RK step. The projection alternative solves for one diagonal
metric component and one diagonal A component to enforce the algebraic
constraints; its independent nonlinear-response check passes, but that does
not provide evolution stability. These maxima are amplification snapshots,
not the first injection location.

The Hamiltonian-relaxation diagnostic adds the residual of `nu chi H_ADM`,
with `nu=0.2h` inside r=M, tapering smoothly to zero at r=1.5M. Its independent
nonlinear directional-response error is 9.09e-16. On a finite Schwarzschild
mass change from M=1 to 1.2, its added source converges away at orders
7.04, 7.31, and 7.09. Nevertheless, a frozen-principal-symbol screen rejects
dt=0.075M (RK amplification 1.196 for a sampled high-frequency mode).
At dt=0.05M the targeted full RK mode remains unstable, with eigenpair
residual 4.49e-10. Passing the smaller-timestep screen did not imply
stability. Original kappa1=0.1/kappa2=0 terms stay unchanged throughout these
diagnostics. No candidate was promoted to C++.

Results are in the local review artifacts `inner-followup-control-results.json`,
`hamiltonian-relaxation-validation.json`, and
`hamiltonian-relaxation-physical-response.json`. The atmosphere/star gate
remains closed until finite-perturbation vacuum stability is established.

The Ricci helper now also exposes the covariant physical `ricci_tensor`.
Its componentwise analytic conformal-metric check and affine tensor-transform
check have maximum errors 3.34e-16 and 7.78e-16. A separate local pointwise
cross-check transforms 200 arbitrary physical metric/K/lapse/shift jets into
conformal variables with Theta=0 and metric-compatible Gamma. Reconstructing
the physical metric and K time derivatives from the geometric Z4c RHS agrees
with independent ADM equations to 1.43e-14 and 1.07e-14. Hamiltonian and
momentum constraints are not imposed on these jets; retaining the Theta RHS
is essential for reconstructing the correct ADM trace evolution. This
supports the nonzero continuum physical response, not the stability of the
spatially discretized evolution. See `physical-ADM-response.json` locally.

An additional offline diagnostic tests a constraint-norm gradient source,
inspired by the functional-derivative construction in
[Tsuchiya, Yoneda and Shinkai](https://arxiv.org/abs/1109.5782). This is not
that paper's BSSN formulation or a claimed reproduction of its results.
For the linear diagnostic, define `C=(h H_ADM, h M_i, C_Gamma^i, Theta)`,
`J=dC/du`, and a nonnegative diagonal mobility W that is zero in the frozen
core and at r>=1.5M. P is the algebraic tangent projector. The added active
source is `-P W P^T J^T delta_C`, with the active/ghost restriction included
in J. Its contribution to `E=||delta_C||^2/2` is exactly a negative square
in exact arithmetic. E here includes derivative-based constraints on all
sampled grid points, including frozen points whose stencils sample active
neighbors; it is a diagnostic norm, not physical gravitational energy.

The independent nonlinear constraint-response, source transpose, and RK
transpose checks have relative errors below 3e-14. The source has zero
support outside the prescribed layer, lies in the linearized determinant/
trace constraint tangent space, and preserves the zero linear residual
exactly. The constraint-free shift response of the source is exactly zero
while the ordinary physical/gauge RHS remains nonzero. These are separate
checks from growth of the coupled evolution.

The source is stiff: its largest decay estimate is about 2055 per unit
strength on this grid, so a strong explicit update would be costly. An
L-stable SDIRK2 source step with strength 5/M is therefore composed with the
unchanged RK3 map using Strang splitting. The resulting diagnostic map is
at most second order in time. Source-only comparison with a matrix
exponential gives local orders 2.89 and 2.95. Independent iterative and
direct implicit solves agree within 2.67e-13 for a complete split step;
restricting the constraint calculation to the affected rows changes it by
2.01e-16. Zero added strength recovers the baseline RK map bitwise. Original
Z4c kappa terms, sixth-order bulk derivatives, KO, outer boundary, core
zeroing, and rate-5 original sponge remain in place. No nonlinear, MPI, GPU,
or mesh-refinement implementation of this experimental layer is claimed.

Both implicit-layer perturbation controls reached their 60M diagnostic
targets normally, with the frozen state exactly zero at every completed
step. Neither is stable. The known-mode seed has late Theta growth
+0.10451240/M over 45–60M; an unrestricted compact random seed gives
+0.10341948/M. Theta grows by factors 22.69 and 22.04 over 30–60M.
Thus a source whose own constraint-norm contribution is nonpositive still
does not stabilize the coupled update. Results and the reviewed figure are
`implicit-constraint-energy-control-results.json` and
`implicit-vacuum-controls.png/pdf` in the local review directory.

At 60M, Theta peaks at (-0.875,0.375,-0.125)M (r=0.960143M) for the mode seed,
and (0.375,0.125,-1.375)M (r=1.430690M) for the random seed. The physical
Hamiltonian residual peaks at r=0.216506M in the frozen core for both cases;
this derivative diagnostic samples neighboring active residuals despite the
core state being exactly zero. Its exterior peaks are at
(-1.625,-1.125,0.375)M (r=2.011685M) and (1.375,0.625,-1.375)M
(r=2.042517M), respectively. The physical momentum-residual norm peaks at
r=0.649519M for both seeds. These are normalized linear snapshots on
rank/block/relative-level zero after step 800, not first-injection locations.
`constraint-energy-profiles.json` separates the core, original sponge,
additional layer, horizon exterior, and outer boundary.

No experimental equation or integrator change has been promoted to the
production executable. The completed exact-zero CPU/MPI/GPU tests remain
valid for the unchanged production numerics, but they are not finite-
perturbation stability passes. No atmosphere, star, or new Aurora production
run was started during these controls.

An independently recomputed full-grid approximate eigenpair of this split
map has growth +0.10463655/M and relative defect 6.94e-8, with unit norm
and exactly zero core before/after the step. This agrees with the late
transient fits; it is not a full-spectrum result. Applying the half-timestep
map to this same vector yields Rayleigh growth +0.09768/M but defect 0.0032,
so the half-timestep number is not a validated eigenvalue or a separate
stability result. See `implicit-mode-full-grid-validation.json`.

### Gauge-only probes separate injection from amplification

The opt-in `problem/vacuum_gauge_pulse_amplitude` hook seeds a compact smooth
pulse in the lapse (`vacuum_gauge_pulse_component=0`) or one shift component
(1–3). Centers `vacuum_gauge_pulse_x1/x2/x3` are relative to the black hole;
`vacuum_gauge_pulse_width` is its support radius. Defaults are amplitude zero,
center (1.75,0,0)M, and width M. The hook requires vacuum with both matter
feedback switches off, an analytic background, the selected gauge residual
enabled, and no simultaneous Theta/characteristic pulse. It changes no
geometric field, Theta, or Gamma and leaves the frozen core untouched. There
are no production equation changes or new synchronization barriers.

```sh
python tst/regression/z4c_gauge_pulse.py \
  --exe /path/to/mpi/athena --output /new/output/directory --ranks 1 4
```

The regression covers all four gauge components, amplitudes 1e-8 and 2e-8,
uniform and refined meshes, and one/four MPI ranks (32 cases). The refined
pulse crosses the actual refinement interface. All passed: initial geometric
residuals remain exactly zero, the geometric response is nonzero and linear
in pulse amplitude, and final active block arrays match bitwise between rank
counts. Theta generation is measured, not required for a pass. Seven invalid
configurations are rejected. The existing 24-case equilibrium/physical-pulse
MPI suite was repeated with the hook disabled; its result dictionaries match
the preceding suite exactly. Separate one/four-thread runs have 26 identical
snapshot payloads per lapse/shift case, including background data. These are
repeatability results, not proof of absence of every possible race. The new
probe hook has not been tested on GPU.

Actual sixth-order C++ controls use the 32^3 box [-4,4]^3 M, dx=0.25M,
dt=0.075M, core radius 0.5M, sponge outer radius M, rate 5/M, and unchanged
kappa1=0.1/kappa2=0. Initial full-minus-background physical ADM Hamiltonian,
momentum, connection, and Theta constraints are exactly zero, including
geometric ghost data. The raw background finite-difference constraints are
not zero. Stage snapshots identify these first Theta injections:

| Seed | First operation | Max absolute Theta RHS | Coordinates (M) | Radius (M) |
| --- | --- | ---: | --- | ---: |
| Lapse 1e-8 | Volume RHS, RK stage 1 | 1.29623378e-10 | (1.125,-0.125,-0.125) | 1.138804 |
| Shift-x 1e-8 | Boundary RHS, RK stage 1 | 3.06094105e-11 | (3.875,0.125,0.125) | 3.879030 |

Both are at cycle zero, rank/block/relative-level zero. The lapse injection
is outside the sponge and inside the horizon. It agrees with
`delta_alpha * H_Theta_RHS_background / 2` to relative error 7.99e-8 (about
1e-17 absolute); this Hamiltonian-like RHS expression is not the independent
physical ADM Hamiltonian. Full-minus-background cancellation at identical
inputs does not cancel a changed lapse multiplying a nonzero discrete
background constraint defect. The shift probe has exactly zero volume and
post-excision Theta RHS at stage one; the first nonzero value is explicitly
captured after the boundary update. Neither location alone establishes the
origin of the later growing eigenmode.

Both controls reach 60M normally, with finite histories and no reported
invalid metrics. They are nevertheless unstable: late (45–60M) Theta growth
is +0.13186503/M for the lapse and +0.13194475/M for the shift. Final Theta
maxima are 2.10563e-8 and 1.70108e-8. Manually injecting Theta is therefore
not necessary to excite the instability. Local artifacts are
`gauge-pulse-injection-results.json`, `gauge-pulse-stage-thread-check.json`,
`gauge-pulse-mpi-regression/results.json`, and
`gauge-pulse-disabled-mpi-regression/results.json`.

An independent flat-space lapse test also isolates a truncation forcing:
using separate sixth-order diagonal D2 and composed mixed D1 derivatives
produces momentum RHS `D1_i sum_{j!=i}(D2_j-D1_j^2) delta_alpha`. Its measured
convergence orders are 5.74 and 5.94. Composing all second derivatives removes
this flat-space discrepancy to roundoff. This does not establish a coding
error or a cure for the black-hole mode. An offline black-hole operator with
composed second derivatives, independently rebuilt background coefficients,
and explicit polynomial derivative-field ghost closure still has a validated
unstable RK eigenpair: +0.08172470/M, relative defect 5.43e-10. An unrestricted
random control reaches 60M with an exactly zero frozen state but also grows.
The candidate requires a wider effective stencil and has not been promoted
to the C++ solver. Vacuum finite-perturbation stability remains unresolved;
the atmosphere/star sequence has not advanced.

The gauge-pulse regression now explicitly matches `debug_balance_freeze/ramp`
to its physical 0.5M/M layer and checks every reported radial maximum's
membership. These audit settings are independent of the evolution settings;
leaving their defaults at 1M/1.4M mislabeled some v1 CSV regions. The initial
injection coordinates and checkpoint profiles above were classified from
coordinates and actual layer radii, so those findings are unaffected. The
regression also asserts exact frozen-core residuals after each recast.

The actual C++ 60M checkpoints put both gauge probes' Theta maxima at
r=0.649519M in the sponge; exterior maxima lie at r=2.011685M. These are
amplification locations, distinct from the initial injection coordinates.
The frozen core remains exactly zero. The reviewed local figure
`gauge-pulse-vacuum-controls.png/pdf` shows time histories, radial shell
profiles, and signed Cartesian slices. Its source data are
`gauge-pulse-spatial-profiles.json`.

A further offline control uses the stationary 1+log Schwarzschild trumpet of
[Bruegmann](https://arxiv.org/abs/0904.4418), distinct from the previously tested
analytic R0=M trumpet. The isotropic horizon radius is 0.830404M. Independent
continuum geometric, ADM Hamiltonian, and ADM momentum checks have maxima
2.23e-15, 2.00e-15, and 8.11e-15; the stationary lapse RHS is below 7e-17.
Tightening the radial-coordinate ODE tolerance changes the sampled state by
1.16e-11. The shift driver still needs background forcing subtraction;
stationary 1+log slicing does not make that driver stationary automatically.

This single-block diagnostic uses dx=0.125M, a 0.125M frozen core, a rate-5
sponge ending at 0.5M, unchanged kappa1=0.1/kappa2=0, full 1+log lapse and
shift-advection response, and a prescribed shift coefficient
`G_bg=0.9 alpha_bg^2 chi_bg` with matching boundary characteristics. The
nonlinear directional-response and RK-transpose checks have relative errors
2.53e-16 and 6.79e-14; the zero linear residual remains exact. An unrestricted
compact perturbation reaches its 60M target normally, with the frozen state
exactly zero throughout. Nevertheless, late Theta growth is +0.07758548/M,
and Theta increases 12.05-fold over 30–60M. Its final maximum is at
(0.3125,0.0625,-0.1875)M, r=0.369755M, inside this test's sponge. This background
alone is rejected as a stability fix. The endpoint Rayleigh estimate has
relative defect 2.28e-5 and is not a validated eigenvalue. No C++/MPI/SMR/GPU
implementation of this background is claimed. Results are in
`stationary-log-background-validation.json`,
`stationary-log-operator-validation.json`, and
`stationary-log-control-results.json` locally.

All 32 gauge-probe cases and seven invalid-input controls passed again with
the corrected audit radii. Their result dictionaries, including final active
block hashes, match v1 exactly. See `gauge-pulse-mpi-regression-v2/results.json`.
A separate MPI first-injection inventory records rank, block, relative level,
coordinates, and RK stage for all four seeds on both meshes in
`gauge-pulse-mpi-first-injection.json`. For example, the refined transverse
shift-y case first develops Theta in stage-two volume RHS at
(3.875,0.625,-0.375)M, rank 1, global block 54, level 1. This is an observation
of the first nonzero Theta, not proof that mesh transfer caused the mode.

For the stationary 1+log candidate, an independently recomputed full-grid
approximate eigenpair has growth +0.07587000/M and relative residual 7.38e-8.
Its state has unit norm and an exactly zero core before and after the step.
Independent physical ADM Hamiltonian and momentum perturbations are nonzero,
so this is not merely growth of a gauge norm. In this dipole mode, Theta and
ADM H peak at (-0.4375,-0.0625,-0.0625)M, r=0.446339M; the momentum norm peaks
at r=0.324760M. These normalized mode profiles are distinct from the random
control's final maximum and from first-injection locations. See
`stationary-log-mode-full-grid-validation.json`. This is a targeted unstable
mode, not a full-spectrum analysis or a production fix.

### Zero-rate sponge semantics

A separate code defect affected the zero-rate control: the legacy
`excision_damp_rate=0` branch multiplied the entire Z4c RHS by the radial
ramp, including physical terms and KO dissipation. Thus reducing the damping
rate to zero selected a different evolution operator instead of removing
relaxation. The zero-rate branch now retains the full annulus RHS and keeps
the same zero RHS in the frozen core. Positive-rate updates remain
`RHS -= rate * (1-ramp) * residual`; kappa1/kappa2 are unchanged. This corrects
a control configuration and does not explain or fix the rate-5 instability.
It also does not change the separate MHD state-projection path.

```sh
python tst/regression/z4c_excision_source.py \
  --exe /path/to/mpi/athena --output /new/source-test/directory --ranks 1 4
```

The regression first reproduced the defect on the preceding executable:
exact vacuum passed, but a compact lapse pulse failed the assertion that
zero damping preserves the annulus RHS. After the fix, all eight cases pass
(rates zero/five, exact vacuum/nonzero pulse, one/four MPI ranks). Every field
is checked before/after the source at all three RK stages. Zero-rate annulus
and exterior RHS arrays are preserved bitwise; the rate-5 additive source
matches its analytic expression to relative error at most 1.07e-16. The core
RHS and post-source residual remain exactly zero. Exact vacuum stays zero,
and all active arrays match bitwise across MPI partitions.

Both CPU OpenMP and MPI builds succeed. Repeating all four single-rank cases
with one/four OpenMP threads gives identical complete snapshot payloads,
including ghosts and background arrays (24 payloads per case); these also
match the one-rank MPI executable. Artifacts are
`zero-rate-source-before.log`, `zero-rate-source-after/results.json`, and
`zero-rate-source-thread-check.json`. This focused source regression does
not establish long-time stability or add GPU coverage for this change.

Four offline controls isolate the roles of zeroing and damping in the
stationary 1+log background test. They retain identical background data,
volume coefficients, gauge, kappa terms, boundaries, KO, projection, and
initial verified mode, changing only the indicated core mask or sponge.
All reach 60M normally; none passes the perturbation-stability gate.

| Change from the stationary-background control | Theta growth, 45–60M (1/M) | Amplification, 30–60M |
| --- | ---: | ---: |
| Release the eight core cells; retain the sponge | +0.07591066 | 9.75 |
| Retain core zeroing; remove sponge damping | +0.10353106 | 22.27 |
| Remove both core zeroing and sponge damping | +0.10191041 | 21.17 |
| Retain zeroing; narrow sponge to 0.125–0.25M | +0.10467619 | 23.06 |

The original targeted eigenvalue is +0.07587067/M. Thus releasing the core
barely changes this mode; removing the sponge increases the observed late
growth. These controls show that neither zeroing nor sponge damping is
necessary for growth in this discretized background. They do not prove that
every earlier Kerr-Schild failure has the same cause. No layer removal is
promoted to a production configuration.

The narrowed layer lies below the radial lapse characteristic horizon at
r=0.303452M; independent scalar-characteristic eigenvalues confirm that all
outward continuum characteristic speeds are negative over its support.
The finite-difference stencil still spans beyond that region, so this does
not establish discrete causality or stability. The default layer reaches
r=0.5M, where the lapse characteristic has an outward branch despite being
inside the physical horizon at r=0.830404M.

Each changed map passes its independent transpose check (relative error
below 4e-15) and exact-zero check. Runs with retained core zeroing keep all
core state components exactly zero at every completed step. The released
cells evolve nontrivially in the other controls. Final Theta maxima occur
at r=0.446339M except when both layer operations are removed, where the
maximum is at r=0.324760M. These are completed-step mode amplitudes, not
first-injection locations. Results and the reviewed comparison plot are
`stationary-log-layer-control-results.json`,
`stationary-log-layer-characteristics.json`, and
`stationary-log-layer-controls.png/pdf` in the local review directory.
