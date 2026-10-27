# Frozen native timestep, elliptic and resolution controls

These controls use the exact qualified static-eight-block production executable
from commit `dcd6bac9df1a2060b656201fd1ec367eccfa4d17`, not a rebuild against later
experimental headers. The executable SHA256 is
`6f4eee9dce9d94daaf57662500afcb975bd98fcdf6947024e6adb7c1edd1a627`.
Its actual compiled-source archive SHA256 is
`eb811a59de9dcd062a39bf6eb646e0e558bd307ed9e4f1a0cd851e5fbe704da3`.

`evidence/native-accuracy-capsule-r2-20261027` contains the executable, all 360
qualified compiler source dependencies, the Kokkos/configuration/static-library
snapshot, independent initializer and matched actual-RK harness. Every PDE
header matches the qualified archive. The harness compiled in 4.00 seconds
using only capsule include/library paths. The 6.90 MB frozen capsule archive and
its packaging manifest preserve these dependencies independently of the live
working tree. Later live formulation/checkpoint edits cannot affect these runs.
The earlier capsule setup failure (older Python tarfile API) occurred before
extraction or native execution and remains recorded separately.

The physical problem is unchanged: isotropic M=1/3 wormhole, Minkowski reference,
selected lapse source and telegrapher parameters, width .24, positive-storage2,
direct signed fields, leading-log enrichment, full variable-metric vector
operator, seven shell radii `.45,.55,.60,.65,.70,.75,1`, future scri and its
prescribed clock/normal/tangential data. There is no new floor, exponent,
parity, source subtraction, shift pin or exclusion.

## Predeclared execution budget and controls

The manifest was written before launch. Each runtime uses Serial CPU execution
with a 180-second process limit. Cases run sequentially; existing jobs and
worktrees are untouched. No new GPU/MPI/remote allocation is used.

| Control | Active cells | dt | Cycles | Elliptic atol |
|---|---:|---:|---:|---:|
| Qualified N6 baseline | 10800 | 1e-4 | 2 | 1e-11 |
| N6 timestep half | 10800 | 5e-5 | 4 | 1e-11 |
| N6 tighter elliptic solve | 10800 | 1e-4 | 2 | 1e-12 |
| N8 related resolution | 25600 | 1e-4 | 2 | 1e-11 |
| N10 related resolution | 50000 | 1e-4 | 2 | 1e-11 |

All cases reach t=.0002. N10's predeclared fallback was one-cycle checkpoint
segments after any reaped timeout; it was unnecessary because the full run
completed in 117.36 seconds. All four cases also pass matched actual Driver RK
controls: all 22 stored coefficients and three shift corrections are bitwise
identical at every recorded cycle. This verifies dispatch against the same
frozen engine, not the independent continuum equations.

| New control | Native full run | Matched actual RK | Initializer-hidden restart |
|---|---:|---:|---:|
| N6 dt/2 | 53.82 s | 48.32 s | 26.81 s |
| N6 atol1e-12 | 35.13 s | 32.84 s | 17.57 s |
| N8 | 61.69 s | 54.82 s | 30.08 s |
| N10 | 117.36 s | 107.81 s | 56.80 s |

All final restart volume/scri snapshots are bitwise identical. The corrected
dt/2 restart additionally verifies complete values/registers/RK/shift/topology.
Its first attempt requested a nonexistent checkpoint by cycle number and failed
before evolution. RST/BIN names use output counters while diagnostic CSV names
use cycles: at output interval .0001, dt/2 produces RST index1 at cycle2 and
index2 at cycle4. The successful correction verifies the sidecar cycle/time
before restoring index1. The original failure is retained; the current runner
selects intermediate checkpoints by sidecar metadata, not filename arithmetic.
The old 15-case regression suite is not rerun per control because the same
production executable already passed it.

## Independent error separation

The two N6 initializers are SHA256-identical to the qualified baseline. The
maximum change over all 25 stored/shift fields is 1.471e-11 for dt/2 and
2.269e-14 for atol1e-12. With baseline Penrose weights, the H difference RMS is
2.352e-12 and 9.867e-14 respectively. These effects are negligible beside the
present short-time spatial/interface/closure defects. This is not a claim about
late-time stability or timestep convergence order.

The comparison retains all active cells, their own Penrose proper-volume
weights, unnormalized L2, point maxima, full-region diagnostics and actual nodal
angular directions/values/scatter. Same-grid controls also use fixed baseline
weights. Masks overlap only for diagnostics; none replaces the all-cell norm.
Derivative-incomplete CSVs cannot reconstruct regional original vector residuals
or momentum/Z component differences, so those are explicitly unavailable.
Global original-equation residuals are retained from the actual native solves.

| Metric at t=.0002 | N6 | N8 | N10 |
|---|---:|---:|---:|
| H volume RMS | .616059 | .509059 | .418473 |
| H unnormalized L2 | 1.65224 | 1.39486 | 1.17556 |
| maximum H | 2.15346 | 1.57085 | 1.13886 |
| M volume RMS | .0710983 | .0283855 | .0137609 |
| Penrose volume quadrature | 7.19289 | 7.50800 | 7.89139 |
| radial beta error RMS versus N2049 | .00102225 | .000516874 | .000238822 |
| maximum actual-node radial beta scatter | .00171138 | .000395733 | .000162246 |
| r<.1 H RMS | 1.76780 | 1.27670 | .931306 |
| Cartesian/shell stencil-neighbor H RMS | .192987 | .0240611 | .00769827 |
| tensor scri pole | 1.44116e-5 | 1.52533e-6 | 3.54611e-7 |

The scri tensor pole fails the unchanged 1e-7 target at all three resolutions;
clock/null/tangential ghost conditions hold at roundoff. At N10 the maximum
original shift RMS over the native run is 9.780e-12 and point maximum 8.071e-10.
Physical constraints and puncture-neighborhood errors remain too large for
scientific acceptance. The minimum sampled radius changes with N and the
proper-volume quadrature changes, so no apparent uniform RMS convergence order
is inferred. Both bulk/interface and puncture behavior are visible in the
regional results rather than excluded. No late lapse-front, wormhole-to-trumpet,
generic changing-regularity or translating-puncture qualification is established.

For cost context only, native full-run shift logs contain total iterations
150/164/191 and summed solve wall times 23.74/48.06/94.10 seconds for N6/8/10.
Warm-start/accepted-state checks are included. This is one Serial configuration,
not evidence of MPI/GPU scalability. Raw per-solve costs/residuals are retained.

## Evidence and reproduction

- `evidence/native-accuracy-controls-20261027`: predeclared budget, exact commands,
  timings, all raw outputs, retained failures and corrected restart proof.
- `evidence/native-accuracy-comparison-20261027`: independent selected-width .24
  N2049/N1025 comparison, every-cell/raw-angular NPZs, JSON and plots. Both radial
  references use classical RK4; native uses actual Ketcheson2S RK4, with dt/2
  providing a separate temporal control.
- `accuracy-regions.png` and `resolution-controls.png`: all-cell errors,
  constraints, angular scatter, changing volume and the unrelaxed scri target.

Lossless gzip copies and their hashes preserve large raw evidence; local
originals remain unchanged. Decompress to original names before running readers.
All capsule sources and its executable are rechecked after execution.

```
python3 tst/elliptic_shift_3d/create_native_accuracy_capsule.py --archive docs/elliptic-shift-3d/evidence/native-static8-build-20261027 --output NEW_CAPSULE
# Capsule creation compiles the matched harness with a 90-second bound.
python3 tst/elliptic_shift_3d/run_native_accuracy_controls.py --capsule NEW_CAPSULE --output NEW_CONTROLS
python3 tst/elliptic_shift_3d/analyze_native_accuracy.py --controls NEW_CONTROLS --output NEW_COMPARISON
```

Use fresh output paths. The exact historically executed runner is preserved in
its evidence directory. The current runner includes the checkpoint-index fix.
The optional `complete_native_accuracy_restart.py` reproduces the historical
follow-up; it is not needed when the current runner already passes that control.
Scientific acceptance remains false.
