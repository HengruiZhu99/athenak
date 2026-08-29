# Ref-GH long-time single-puncture robustness and closed-loop q qualification

## Final classification

`NOT ESTABLISHED`

The first mandatory Phase-2 trajectory—the exact matched stationary trumpet at
medium resolution, q=1, with the q controller disabled—fails at
`t=1.123484M`.  A second run with a radically different MPI decomposition
reproduces the same failure at the same cycle and time with conditioned history
agreement of `1.573e-13`.  The controlling conditional gates therefore stop
the campaign before the resolution ladder or any moving/controller cases.

This result is specifically a failure of the frozen Ref-GH implementation to
pass the requested robustness campaign.  The evidence does not uniquely
distinguish a mathematical formulation defect from an equation-implementation
defect.  It does exclude the extreme one-MeshBlock-per-rank decomposition as
the sole cause.

## 1. Scope and immutable implementation

- Repository: `HengruiZhu99/athenak`
- Branch: `codex/ref-gh-single-puncture-robustness-20260829`
- Exact starting commit: `53f70903c82f2f8670df7aad6aa14e7ef646ad82`
- Frozen production implementation: `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`
- Kokkos submodule: `6739bc623081648af9e752b616d9671527922cbf`

At every campaign commit, `src/` remained byte-identical to `a09caf70`.  The
only additions were a source-external input, health/comparison analysis tools,
Aurora launch plumbing, compact evidence, and this report.  No equations,
controller gains/bounds, damping parameters, finite-difference operators,
time integrator, boundary algorithm, or production kernel were changed.

The retained system is the analytic radial-q backend with compatible
first-order GH equations, the existing gauge driver and reference subtraction,
gamma0=gamma2=1, FD4, RK4, CFL 0.05, dissipation 0.02, the existing q estimator
and controller parameters, and exact projected stationary-trumpet physical
boundaries.  Fluid coupling, p control, wormhole-to-trumpet evolution, moving
center, SMR/AMR, new boundary conditions, and binaries remained out of scope.

## 2. Phase 0 regression freeze

The fresh local Release/Serial source-unit run passed without tolerance
changes.  Its executable SHA-256 was
`d1e335849b01cfb2b2bf0ddad2e8e5ee3d962023b3d2068018d95ed87458f49b`.
Key gates were:

| Gate | Result |
|---|---:|
| analytic coefficient oracle | 216 samples, `8.882e-15` max error |
| expanded radial oracle | 2160 samples, `1.488e-13` conditioned error |
| generated geometry oracle | 2376 samples, `2.331e-15` error |
| moving mixed-jet gauge oracle | 2160 samples, `1.248e-14` motion error |
| compact boundary projection oracle | 2160 samples, `4.565e-14` metric error |
| all-61 RHS oracle | 4320 samples, `2.842e-14`, compatible and standard Phi |
| production cache oracle | `1.638e-14` conditioned scaled Linf |
| exact Minkowski cycle-zero check | zero error |

Two independent deterministic SymPy regenerations were byte-identical to each
other and to the committed generated geometry/gauge/source headers.

Aurora debug job 8790831 then passed the bounded production PVC gate on node
`x4705c2s0b0n0`.  Eight ranks mapped to distinct PVC tiles 0.0 through 3.1;
the one-tile and eight-tile evolved dynamic-q histories agreed to
`3.8898e-14` against the frozen `5e-12` tolerance.  Kokkos selected SYCL/Level
Zero, GPU-aware MPI was enabled, the production image allocated zero generic
reference-cache bytes, and PBS exit status was zero.

Phase 0 is therefore passed.  Its one-cycle result is not long-time evidence.

## 3. Phase 1 accepted-state health telemetry

The source-external binary64 analyzer records the requested relative-metric
signature/eigenvalue/condition data, relative and physical lapse extrema,
maxima of Psi and its inverse, native GH/reduction/curl RMS and Linf, physical
metric/lapse/shift errors, and offline q-estimator statistics.  Every derivative
constraint norm discards the complete FD4+dissipation stencil-overlap cube
around the puncture.  These values never feed the q controller.

Predeclared fail-only thresholds are native constraint RMS 1, Linf 10,
q in [0.5,2.5], and |qdot| <=0.25, in addition to finite-state and
SPD/Lorentzian-signature requirements.  A cycle-zero local plumbing smoke
passed at roundoff.

Both Phase-2 runs failed before the 2M binary64 cadence.  Consequently their
offline health summaries contain only the initial state.  `all_pass=true` in
those JSON files describes t=0 only and is not a trajectory result.  The live
0.2M histories provide the positive-time failure evidence below.

## 4. Phase 2 matched-medium configurations

Both attempts used the exact physical stationary unit-mass n=2 trumpet and an
exactly matched q=1 analytic reference, with the controller and prescribed-q
path disabled.  The physical grid was uniform 96^3 on `[-2M,2M]^3`, h=M/24,
with 216 16^3 MeshBlocks and four ghost cells.

| Job | Queue/nodes | Ranks | Blocks/rank | Requested | Result |
|---:|---|---:|---:|---:|---|
| 8790836 | debug-scaling / 18 | 216 | 1 | 100M | invalid timestep at 1.123484M |
| 8790840 | debug / 1 | 12 | 18 | 100M | invalid timestep at 1.123484M |

Both used the same PVC executable, SHA-256
`b247762c83e5bba2b8a5331f9ce372d17462eb6f7ef59c63b4ddef39306a05e0`.
Each mapped every rank to a distinct PVC tile, used GPU-aware MPI, and exited
143 after the fail-closed invalid-effective-timestep guard.  Neither reached
the first 2M checkpoint; only its t=0 restart exists.

## 5. Numerical failure evidence

The two runs have the same six accepted history times from t=0 through
t=1.000922M, then both reach cycle 330 at t=1.123484M and fail.  Across Ref-GH,
controller, and six common-ADM streams, the global conditioned Linf difference
is `1.572963e-13`, below `5e-12`.  Controller histories are identical.

For the 216-rank run, selected puncture-stencil-excluded/live diagnostics are:

| t/M | GH RMS | reduction RMS | curl RMS | metric-error RMS | Q Linf | source-frame Linf |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `5.91e-14` | `1.08e-14` | `1.32e-14` | `3.53e-15` | `3.20e-11` | `7.46e-11` |
| 0.4017 | `4.03e-11` | `8.56e-13` | `1.20e-11` | `4.74e-12` | `1.13e-5` | `3.19e-5` |
| 0.8001 | `1.64e-6` | `2.90e-8` | `5.06e-7` | `4.46e-8` | `5.49e-1` | `1.46` |
| 1.0009 | `3.60e-4` | `6.37e-6` | `1.10e-4` | `9.74e-6` | `1.26e2` | `4.87e3` |

Log-linear fits over t>=0.2 give e-folding times of 0.0376M for GH RMS,
0.0451M for reduction RMS, 0.0367M for curl RMS, and 0.0337M for the
source-frame maximum.  This is fast exponential growth, not slow secular
roundoff accumulation.

At the last history point the recorded relative-metric condition number is
still `1.0000487` and `bad-state=0`, but the regular-field maximum is 83.2,
physical lapse remains within [0.01785,0.75035], and physical lapse/shift RMS
errors are `3.16e-8`/`1.13e-8`.  The timestep becomes invalid before the next
accepted history record.

The native near-region GH/curl sums account for essentially all corresponding
global growth, while the common ADM 2<=r<4 diagnostics remain nearly unchanged.
This associates the observed instability with the inner region, but the 2M
binary output cadence produced no positive-time field snapshot.  The evidence
therefore cannot distinguish a puncture-localized mode from a broader r<2 mode,
nor identify an exact grid point or source term as causal.

The boundary is not a persuasive explanation for the onset.  With outer faces
at coordinate distance 2M and maximum characteristic speed about 0.612, a
simple face-to-r<1 arrival estimate is about 1.63M; the run fails earlier at
1.12M.  This is supporting timing evidence, not a rigorous domain-of-dependence
proof for every characteristic family.

## 6. Performance observation and profiling decision

Excluding setup, 330 cycles contain 291,962,880 active zone-cycles.  The
216-tile attempt took 22.43 s (`1.302e7` zone-cycles/s), while the 12-tile
attempt took 72.45 s (`4.030e6` zone-cycles/s`).  Eighteen times as many tiles
provided only 3.23x speedup, about 17.9 percent parallel efficiency for this
short shape.  Thus the intended one-block-per-tile production scaling is poor
even aside from the numerical failure.

No profiler was launched.  The controlling goal freezes production `src/` and
explicitly forbids resuming performance optimization; more importantly, the
mandatory numerical trajectory is invalid before 2M.  Profiling an invalid
trajectory would not advance the robustness qualification.  The scaling
measurement is recorded as an observation, not a source-level bottleneck
diagnosis or optimization result.

## 7. Conditional phases not executed

The goal permits the resolution ladder only if the first medium q=1 run is
finite and bounded.  That condition is false.  Therefore:

- the h=M/16,M/24,M/32 t>=20M ladder and retained high-resolution t=100M run
  were not executed;
- static q=0.9/1.1 and q=0.75/1.25 cases were not executed;
- prescribed C2 q pulses were not executed;
- closed-loop q relaxation and finite-resolution q-equilibrium convergence
  were not executed;
- restart equivalence across all 61 fields/q/histories was not executed;
- separate Phi and gauge-driver damping perturbation tests were not executed.

No claim is made for long-time stationary robustness, resolution convergence,
moving references, q control, restart continuity, damping robustness, or
production performance.

## 8. Artifacts and Aurora state

Compact evidence is under
`artifacts/ref_gh_single_puncture_robustness_20260829/`:

- `phase0_local/`: deterministic regeneration, complete source-unit transcript,
  and telemetry smoke;
- `phase0_aurora_8790831/`: PVC mapping, build/configuration provenance,
  one/eight histories, logs, and comparison;
- `phase2_attempt_8790836/`: 216-rank failure histories, mapping, log, status,
  initial health record, PBS record, and hashes;
- `phase2_discriminator_8790840/`: 12-rank reproducer with the same evidence;
- `phase2_decomposition_comparison.json`: quantitative cross-decomposition
  agreement and growth fits;
- `aurora_jobs_final.txt`: final campaign job enumeration.

The uncommitted large restarts and binary outputs remain beneath:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_single_puncture_robustness_20260829_253c78e2/runs/`

The campaign used only jobs 8790831, 8790836, and 8790840.  At wrap-up,
`qstat -u hzhu` reports no queued, held, running, or exiting jobs.  No other
user's allocation, job, or directory was touched.

## 9. Required conclusion

`NOT ESTABLISHED`

The analytic backend remains oracle- and bounded-PVC-qualified, but the frozen
matched q=1 stationary trumpet develops a decomposition-independent exponential
inner-region instability and fails before 1.2M.  This directly contradicts the
first long-time robustness requirement, so none of the downstream q-control
claims can be made.
