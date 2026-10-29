# Thin-overlap prototype: preliminary evidence

Experimental branch `codex/hyperbolic-thin-overlap`, based on
`e48a8c43f` of `codex/hyperbolic-core-shell`. The V11 default remains a control.
See [design and derivation](DESIGN.md) and the [user specification](request.txt).

Implemented: separate Cartesian extension count; an explicit shifted
six-point donor fallback; paired weighted-adjoint penalties on common value
and normal-derivative traces; batched cached device application; two-interface
diagnostics; an isolated penalty timestep bound; and configuration-bound V12
restart. The volume remains cell centered, avoiding a new puncture node at
r=0. Spherical shells remain spherical. Actual VC MeshBoundaryValues classes
are inspected infrastructure, not wholesale dependencies of this prototype.

## Preliminary results

| Check | Result |
|---|---|
| Centered donors, n12, one/two layers | Rejected: missing donor coverage |
| Centered donors, n16, one layer | Rejected: missing donor coverage |
| Centered donors, n16, two/three layers | Construct successfully |
| Explicit shifted donors, one layer, n12/16/24 | Construct successfully, no outside-hull extrapolation |
| Positive-weight penalty energy identity | Relative discrepancy <2e-15 in tested random data |
| Constant trace mismatch | <1.2e-14 |
| CPU/device physical penalty RHS | Maximum difference 5.42e-19 |
| Partitioned trace norms | Equal to whole-domain norms in the fixture |
| Minkowski, 1/2/3 layers, kappa=0/1 | Four steps preserve state to 3.94e-15, no floor interventions |
| Three layers, zero penalty, shifted donors off | V11 checkpoint payload bitwise identical |
| Nonspherical short evolution | One/two ranks and restarted checkpoint bitwise identical |
| Restart with different penalty strength | Rejected |

At n16 the one-layer configuration has 143,632 evolved points, compared with
159,200 for the control: 9.78% fewer. Shell angular and radial extensions are
unchanged. This is not elimination of geometric overlap, nor an accuracy-
matched speedup. The initial serial CPU measurements show additional penalty
work; no GPU speedup is established.

![Preliminary checks](preliminary.png)

The smooth-data plot measures interpolation consistency only. These short
Minkowski/oblique tests are not interface-crossing, long-time stability,
puncture, or convergence qualification. The `3M` result in the neighboring
V11 report belongs to the preserved control, not this variant. No horizon
acceptance tolerance has changed.

Reproducible local preflight:

```sh
cmake -S . -B build-thin-overlap -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON
cmake --build build-thin-overlap --target \
  hyperbolic_shift_3d_paired_trace hyperbolic_shift_3d_paired_trace_device \
  hyperbolic_shift_3d_mpi_thin_trace hyperbolic_shift_3d_mpi_continuous_positive_fast -j2
build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_paired_trace 16 1 1
build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_paired_trace_device
python3 tst/hyperbolic_shift_3d/test_thin_trace_mpi.py \
  --binary build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_thin_trace \
  --control build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_continuous_positive_fast \
  --mpiexec /path/to/mpiexec --output /fresh/output/directory
```

Runtime variables, only for the experimental target:
`HYPERBOLIC_TRACE_LAYERS` (1..3, default 1),
`HYPERBOLIC_TRACE_SHIFTED` (0/1, default 1),
`HYPERBOLIC_TRACE_KAPPA` (nonnegative, default 0), and
`HYPERBOLIC_TRACE_NORMAL_RATIO` (nonnegative, default .25).
The extra `oblique` initial condition is a compact off-center lapse pulse;
its continuous evolution is not an exact analytic solution.

## Bounded oblique probe and sparse integration

The prototype source is `b9685c413`; the following longer probe uses the same
frozen executable as the first preflight. All three cases completed 80 steps
to coordinate t=.04 with dt=.0005, n12, and identical diagnostics:

| Variant | Native seconds | H L2 at .04 | M L2 at .04 |
|---|---:|---:|---:|
| Three-layer control | 235.91 | .00442054 | .00922456 |
| One layer, zero penalty | 210.38 | .00442873 | .00926886 |
| One layer, kappa=1, eta=.25 | 240.82 | .00440651 | .00914741 |

All retained null errors <=1.12e-15 and used zero lapse-floor interventions.
The added penalty used 28.73 seconds. Its small constraint reduction has not
shown enough benefit to offset that cost on this short CPU case. These are
exploratory local timings with other local work present, not a dedicated GPU
benchmark. The pulse is off-center and evolves through the interface region;
reflected characteristic amplitudes and independent time/space convergence
remain unmeasured. The three-layer result is a coarse control, not truth.

![Matched oblique probe](oblique-probe/comparison.png)

`oblique-probe/comparison.json` also records volume-weighted lapse, chi,
radial-shift and tangential-shift differences on common nominal sample
coordinates. The diagnostic partition and cutoff are the same. Trace mismatch
plots have different physical Cartesian trace locations as layer count changes;
their absolute values are not a common-location error comparison.

The separately committed sparse work `fd522df9248c4911c9352057156b3ec58e7dc02b`
is integrated through a distinct `hyperbolic_shift_3d_mpi_thin_trace_sparse`
target. Its dependency union includes trace interpolation AND adjoint-lift
support, in addition to bulk, sampled-normal, scri and diagnostic dependencies.
Poisoned two-rank oblique, restart and short puncture runs reproduce complete
non-sparse checkpoint bytes. The puncture fixture reaches only .0015M, not 3M.
Its initial data agree exactly with all 143,632 corresponding V11 samples.

The Aurora controller now binds each thin geometry/penalty configuration and
whitelists per-case environment variables. It rejects stale V11 or changed
V12 admission parameters. Worker tests were updated for the new explicit
environment argument; the initial mock-signature failure is retained beside
the repaired controller tests. All 106 controller/preflight tests subsequently
passed. No physical acceptance threshold was changed.
No GPU job has been submitted from this experimental fork yet.

## Remaining gates

1. Longer oblique and spherical interface probes, reflection/error diagnostics
   and independent timestep refinement. Preserve failed cases.
2. Qualify the integrated sparse trace target on all 12 GPU ranks; local
   poisoned two-rank trajectory equivalence has passed.
3. Fresh frozen SYCL+MPI source/build and a 12-tile, single-debug-node preflight,
   sharing the existing campaign allocation/storage guards.
4. Isotropic puncture through at least 3M, with matching 1D norm conventions
   and accepted horizons only. The n16 thin initial data have been generated,
   with provenance retained; a two-step equivalence fixture passed, but no
   successful duration test is claimed.
5. Comparable-accuracy performance and convergence comparison; AMR evolution
   remains a separate unresolved infrastructure limitation.

Recommendation pending these gates: retain V11 as the control. One-layer
coupling is a viable bounded experiment only with an explicit donor policy;
its construction and dissipative added term alone do not justify adoption.

### GPU qualification preparation

`aurora/thin_plan.py` generates a separately bound 12-tile preflight for the
V12 variant. It checks paired-lift CPU/device parity on each tile, short
Minkowski preservation for Cartesian layers 1/2/3 and kappa 0/1, an oblique
full-gather versus poisoned sparse restart comparison, and a 100-step puncture
comparison through 0.15 M. The V12 checkpoint auditor verifies the 184-byte
header, point count, log-variable configuration, and exact trace parameters;
it rejects geometry or penalty mismatches before accepting equivalence.
This plan is preparation, not GPU evidence until a completed collected run
is recorded below. It does not qualify the requested 3 M puncture evolution.

```sh
python3 tst/hyperbolic_shift_3d/aurora/thin_plan.py --output plan.json
```

The proposed frozen source/build names are `source-thin-v1` and
`build-sycl-mpi-thin-v1`, separate from the ongoing control/performance work.
