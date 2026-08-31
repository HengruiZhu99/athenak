# Ref-GH moving-reference versus hard-freeze discriminator

## Scope and claim boundary

This report records a bounded, one-resolution discriminator on Perlmutter for
the STANDARD-Phi, relative-damped, fixed-core Ref-GH single-hole trajectory.
It compares two continuations from the same t=2M moving-reference checkpoint:

1. the prescribed reference motion continues; and
2. the complete reference is hard-frozen at xi=0.25 with xi_dot=xi_ddot=0.

No physical gauge target, GH equation, finite-difference stencil, RK scheme,
KO coefficient, damping parameter, or blend profile was changed.  This is not
a resolution study, a long evolution, or evidence of a stable/convergent
trumpet.

## Implementation and audit

The implementation adds a `hard_freeze` continuation mode and makes the
controlled reference static after the fork.  Restart reprojection reconstructs
the physical coordinate metric and its first derivatives using the pre-freeze
moving reference, then expresses the same data in the frozen frame.  At t=2M
the reprojection reports:

```text
xi=0.25
old xi_dot=0.125/M, old xi_ddot=0
new xi_dot=0, new xi_ddot=0
spatial-state change Linf=2.22045e-15
Pi change Linf=1.00634
```

The nonzero Pi change is the expected representation change when the reference
frame velocity is removed; it is not hidden clipping or a state reset.

The source-unit oracle samples five activation values and seven radii.  It
checks 35 hard-frozen reference instances and finds exact binary64 zeros for
all reference temporal jets, time invariance, moving-frame derivatives,
connection time derivatives, gauge-reference temporal terms, and related
quantities.  The full source-unit result on an A100 also retained the prior
flat and nonflat source tolerances (`5.55e-17` and `3.89e-16`).  Runtime
max-location diagnostics independently report zero `reference_dt_frame` and
zero `reference_dt_connection` throughout the frozen branch.

To compile and run the unchanged mathematics with NVCC/Cray MPI, two
performance-portability corrections were necessary:

- nested CUDA extended-lambda launch bodies were refactored into public/free
  templated launch helpers without changing their device expressions; and
- each process selects its Slurm-local CUDA device before `MPI_Init`, allowing
  Cray GPU-aware MPI and Kokkos to initialize the same device while retaining
  all four node-local GPUs in the visible namespace.

The final executable links `libmpi_gtl_cuda.so.0`.  All 16 ranks map to distinct
A100 UUIDs, four per node, without the earlier MPI device-switch or CUDA IPC
errors.  The failed build and launch-binding attempts are preserved but are
not used as numerical evidence.

## Run provenance

- Branch: `codex/ref-gh-relative-damped-single-hole-20260830`
- GPU-run source: `823f851d184185565fb4046927754810b551e24b`
- Allocation: Perlmutter `57779143`, account `m3328_g`, QOS
  `gpu_interactive`
- Nodes: `nid001212`, `nid001213`, `nid001216`, `nid001217`
- GPUs: 16 NVIDIA A100-SXM4-40GB, one MPI rank per GPU
- CUDA/Kokkos/MPI: CUDA 13.2, Kokkos 4.7.2 CUDA+Serial/AMPERE80,
  Cray MPICH 9.1.0 with CUDA GTL
- Grid: `[-24M,24M]^3`, 32^3 active cells/MeshBlock, 328 MeshBlocks,
  3 physical refinement levels, finest `h=1/16M`
- Numerics: FD4, RK4, CFL 0.05, KO 0.02, gamma0=gamma2=1
- Gauge: relative-damped, gauge driver disabled, reference-gauge subtraction
  disabled
- Puncture: between cells; all diagnostic samples whose complete FD/KO stencil
  can overlap the puncture are excluded
- Fresh common seed: t=0--2M; both branches: t=2--4.2M
- Scheduler result: allocation completed normally in 01:36:09 and was
  relinquished; no further job was launched

The actual startup tree was 328 MeshBlocks: 208/56/64 on physical refinement
levels 1/2/3.  The `num_levels` and region keys are reported as unused by the
generic input-unused audit, but the startup tree and `h=0.0625M` native-cell
diagnostic confirm that static refinement was constructed.

## Established facts

At the common fork state, t=2M:

| quantity | value |
|---|---:|
| xi, xi_dot | 0.25, 0.125/M |
| GH RMS | 1.05902e-4 |
| reduction RMS | 1.74349e-5 |
| curl RMS | 3.09458e-4 |
| e_G, e_alpha | 2.46839e-2, 3.75124e-2 |
| bad-state count | 0 |

Both branches reached t=4.2M with finite reported state and `bad-state=0`.
The complete endpoint comparison is:

| quantity | continued motion | hard freeze |
|---|---:|---:|
| xi, xi_dot at t=4.2M | 0.525, 0.125/M | 0.25, 0 |
| GH RMS | 4.13903e-3 | 1.26001e-2 |
| reduction RMS | 9.807996e-4 | 2.994740e-4 |
| curl RMS | 1.278516e-2 | 2.785922e-3 |
| e_G | 2.08880e-1 | 1.36999e-1 |
| e_alpha | 3.38505e-1 | 1.48592e-1 |
| relative metric condition max | 4.43296 | 2.66818 |
| relative v^2 max | 7.37508e-2 | 1.94276e-1 |
| bad-state count | 0 | 0 |

The hard freeze introduces an immediate GH-gauge mismatch.  At the first
post-fork history sample, t=2.020653M, GH RMS is `1.95201e-2`, versus
`1.09205e-4` for continued motion.  Reduction and curl are still matched there
to within about 0.2%.  After this impulse, hard-freeze GH decreases to
`1.26001e-2`; it does not exponentially grow during the observed dwell.

Log-linear fits over t=2.15--4.2M give:

| sector | continued slope /M | hard-freeze slope /M |
|---|---:|---:|
| GH RMS | +1.6115 | -0.18945 |
| reduction RMS | +1.7466 | +1.1853 |
| curl RMS | +1.6019 | +0.95789 |
| Pi RHS Linf | +1.0921 | +0.24479 |
| frame-correction source | +1.3757 | +0.50617 |

At t=4.2M the moving branch has 3.28 times the hard-freeze reduction RMS and
4.59 times its curl RMS.  Its Pi/Phi RHS maxima are 13.36/8.75, versus
1.87/0.886 for hard freeze.  Moving-branch GH, reduction, curl, Pi RHS, and the
largest nonlinear source pieces localize around r=0.47--0.56M, coincident with
the fixed-core blend/window-gradient region rather than the outer boundary.

The hard-freeze run is about four times faster (2.17e7 versus 5.22e6 active
zone-cycles/s) because the fixed reference uses the static cache path.  This is
an implementation performance observation, not a physical conclusion.

## Numerical evidence and interpretation

The data reject the strongest form of “the same instability is entirely
unchanged after reference motion stops.”  Removing all reference-time
derivative sectors reduces the observed reduction/curl growth rates and the
final RHS/source maxima substantially.  Continued blended-reference motion is
therefore a material amplifier of the unstable numerical/formulation sector.

The data do **not** establish that ongoing motion is necessary.  The frozen
branch retains positive, well-fitted reduction and curl growth, with e-folding
times about 0.84M and 1.04M over this short window.  It also starts with a large
GH constraint impulse caused by discontinuously changing the moving-reference
gauge/frame state.  Consequently, a clean comparison of GH growth or a claim
that the frozen intermediate operator is stable would be invalid.

The power diagnostics also do not justify a single pure-power explanation.
At t=4.2M the hard-freeze mean Delta-q is -0.055 in the stencil-safe inner
shell, -0.062 in the blend shell, and +0.123 outside the blend, with substantial
shell variance.  The moving branch has a different spatial pattern.  These
radially blended states are not globally described by one q exponent.

## Remaining hypotheses

The present evidence is consistent with more than one mechanism:

1. reference-time-derivative/source terms drive much of the rapid moving-branch
   growth near r~0.5M;
2. the fixed intermediate blended operator itself has a slower unstable
   reduction/curl mode; or
3. the moving trajectory excites a mode before t=2M which persists after
   freezing, with the abrupt freeze additionally injecting a GH-gauge impulse.

The matched pair cannot distinguish hypotheses 2 and 3.  Correlation and
radial co-localization are evidence, not a causation proof.

## Smallest next discriminator

No direct fixed-intermediate run was launched.  It is not inexpensive in the
current problem generator: fresh wormhole physical data are initialized in the
wormhole reference, so starting with the xi=0.25 fixed reference requires a
careful one-time representation projection and a gauge-consistent initial
state rather than an input-only override.

The smallest decisive next test is a bounded fresh direct-fixed xi=0.25 run,
using exactly the same physical wormhole data, grid, STANDARD-Phi ordering,
gauge, and numerics, with the initial physical first derivatives projected into
the fixed reference.  Before using its result, an oracle must verify matched
physical metric/derivatives and quantify its initial GH constraint.  If that
setup cannot avoid a gauge impulse, use a short smooth-stop fork whose endpoint
has xi_dot=xi_ddot=0 and compare only after the stop transient has exited the
blend region.  No parameter tuning or resolution ladder should precede that
discriminator.

## Evidence locations

Compact evidence is committed under
`artifacts/ref_gh_reference_motion_freeze_perlmutter_20260831/`.  Large
checkpoints remain at:

`/pscratch/sd/h/hzhu/refgh-reference-motion-freeze-20260831.7Y5n8O/reference_motion_freeze_57779143`

The common t=2M and both final t=4.2M checkpoint paths, sizes, and SHA-256 hashes
are recorded in `restart_manifest.tsv`.  No large checkpoint or field
dump is committed.
