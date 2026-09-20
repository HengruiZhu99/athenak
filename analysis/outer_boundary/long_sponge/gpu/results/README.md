# GPU sponge controls: verified results and incomplete cached evidence

**A wider sponge alone did not remove the instability.** With `kappa1 = 0.1`,
the baseline failed at 5650.2 M and the wide-sponge case failed at 6420 M.
The matched `kappa1 = 0` lapse-pulse case reached its actual 20000 M target
with valid final checkpoints. Direct-Theta tests were then submitted to test
constraint response above the near-roundoff amplitudes of that lapse control.
Their final results are **not available in this package**.

These are independent flat-background vacuum diagnostics, not the TDE run,
a black-hole interior, matter evolution, or AMR validation. No production job,
input, checkpoint or monitor was changed. Here M is the reference production
mass unit; these boxes themselves have `bh_mass = 0`.

## Actual stopping and collection status

| Job and case | Configuration | Verified outcome or available evidence |
|---|---|---|
| 8842172 / baseline | No sponge; `kappa1 = 0.1`, `eta = 2`, lapse damping 0.1 | Invalid-state failure at **5650.2 M**, cycle 9417 |
| 8842172 / radial_k01 | Wide radial sponge; same damping | Invalid-state failure at **6420 M**, cycle 10700 |
| 8842171 / radial_k0 | Same sponge; `kappa1 = 0`, `eta = 2`, lapse damping 0.1 | **Reached 20000 M**, cycle 33334; application and PBS exit 0; all eight final checkpoint files passed |
| 8842248 / theta_primary | Direct Theta amplitude 1e-6; `kappa1 = 0`, `eta = 0.02`, lapse damping 0.01 | Cached full histories through **30800 M**; a later status-only sample reached **45571.2 M**. Final stop and checkpoint are unverified |
| 8842248 / theta_lapse01 | Fresh direct Theta amplitude 1e-6; lapse damping 0.1 | Exact submitted input available; no collected evolution output |
| 8842283 / theta_amplitude | Fresh matched primary with Theta amplitude 1e-7 | Submission and exact input available; no collected evolution output |

The last successful remote status check was at 19:07:52 UTC on 2026-09-20.
A fresh collection attempt at 20:28 UTC failed because the Aurora control
socket was gone and direct BatchMode SSH reported
`Permission denied (keyboard-interactive,hostbased)`. See
[collection-access.json](collection-access.json). Elapsed calendar time is not
evidence that C or D reached their 50000 M targets. The old scheduler state
in [status-latest.json](status-latest.json) is historical, not a current query.

A/B's two failures were independent fresh starts and were not restarted.
Their wrappers recorded exit 143 after MPI abort, and PBS job B exited 1.
Both had valid earlier checkpoints at 5000.4 M; those files do not validate
the failed endpoints. These failures are not walltime stops. A completed in
2147.4 application seconds, not by exhausting its allocation. The available
three-cycle zero gates passed exactly, including all eight rank files and
metric ghosts; the uncollected C ablation and D gates are not declared passed.

## What the completed comparisons show

Over 4000–5000 M, whole-domain Theta RMS grew exponentially with
`gamma = 0.006704/M` without the sponge and `0.005747/M` with it
(R² 0.999998 and 0.999989). Widening delayed the failure but did not cure it.
At 5000.4 M, respectively 99.684% and 99.707% of the proper-volume Theta²
lay within 256 M of physical faces. Active Theta peaks were at
(-32, 224, 2016) M and (288, 224, 2016) M, 32 M from the +z face. These
are locations of late amplification, not the first injection.

The first printed primitive failures were fourth physical ghost cells at
(32, 32, 2272) M without the sponge and (2272, 96, -1952) M with it.
Their determinants were negative. Separately recorded active-cell fatal
checks occurred later; full excerpts preserve this distinction in
[failure-excerpts](failure-excerpts). Internal tangential ghost indices do
not make these points physical-domain corners.

The `kappa1 = 0` case ended at 20000 M with maximum |Theta| 2.129e-14 and
whole-domain RMS 5.835e-16. All saved payload fields and all ghost metrics in
the final eight-rank cohort were finite and positive definite. Its retained
geometry decreased between 5000.4 and 20000 M: maximum |delta chi| fell from
1.970e-9 to 6.501e-10, and the coordinate-Frobenius conformal-metric residual
from 1.233e-9 to 3.662e-10. Near-noise Theta does not establish general
perturbation stability.

![Completed lapse-pulse comparisons](long-growth-comparison.png)

## Direct-constraint test: incomplete evidence

The regular direct seed is
`Theta = A exp[-r^2/(2 * 384^2)]`, with A = 1e-6 in C and 1e-7 in D.
Unlike the lapse control, this Gaussian has nonzero boundary tails;
30.93% of its initial proper-volume Theta² overlaps the sponge ramp.
The small late plateau must not automatically be called machine noise.
A retained incoming characteristic is one hypothesis requiring the separate
boundary-state and amplitude-scaling audits. D's missing final data cannot
be used to claim linear scaling.

At the end of C's cached histories (30800 M), maximum |Theta| was
4.5840e-13 and exterior RMS was 1.25894e-13. The later status-only sample
at 45571.2 M reported maximum |Theta| 4.58515e-13 and exterior RMS
1.25833e-13. These samples support a small bounded interval in the available
data; they do not prove long-time saturation, completed 50000 M evolution,
or validity of every intervening checkpoint. No final C/D growth or
lapse-damping comparison is claimed.

A history-only radius-512 M mask splits diagnostics without changing the
fields. `Theta-norm / Volume` is **exterior** RMS²; `Theta-int2` is the
**core** Theta² integral. Core curves plot sqrt(core integral), not core
RMS. Whole L2 is sqrt(core plus exterior integrals). This diagnostic mask
performs no physical excision. C profiles stop at 20000 M in the cached
checkpoint data; its histories extend farther.

![Cached primary direct-Theta evolution only](theta-comparison.png)

## Exact setup and reproducibility

All jobs used one Aurora node, eight GPU MPI ranks, allocation `MHDTidal`,
and one-hour PBS limits. The immutable repaired executable SHA256 is
`a6c3af79571819fba5dc2252ceb9abacb31279feec440f2c542e342dfee43639`.
The cube is [-2048, 2048]³ with 64³ active cells, eight 32³ blocks and
spacing 64 M. It evolves the residual spacetime with no forced metric
override and no matter feedback. All use sixth-order volume differences,
G1 background-adapted gauge, the original `zero_rate` boundary, linear ghost
extrapolation and KO coefficient 0.5. The experimental radiation boundary
is not selected in these jobs.

The radial sponge is zero inside 512 M and rises with the C² quintic profile
`s^3 (10 - 15 s + 6 s^2)`, where `s = clamp((r - 512)/1280, 0, 1)`.
Its rate reaches 0.001/M at 1792 M, before the nearest face at 2048 M.
It continuously relaxes residuals and remains a mitigation under test,
not a demonstrated constraint-preserving boundary closure.

A/B use dt = 0.6 M and a compact C-infinity lapse bump of amplitude 1e-8,
center (128, 64, 0) M and support radius 256 M. The bump is
`A exp(1 - 1/(1-q^2))` for q < 1 and exactly zero otherwise; its support
lies within r < 399.11 M, entirely outside the layer. C/D use dt = 3.2 M,
following short timestep comparisons, and the Gaussian Theta seed above.
C's two independent application caps are 28 and 25 minutes; D's is
55 minutes. They are caps, not achieved evolution times.

[Exact submitted inputs and PBS files](jobs) preserve the actual output
cadence: A/B restart every 5000 M; C/D restart every **1000 M**, full Theta
binary output every 64 M, and constraint binary output every 128 M.
Binary and restart output are per rank. Original hardcoded remote paths in
these records identify the actual jobs; these archived PBS files are not a
portable submission tool and should not be rerun blindly.

The compact package contains lossless numerical history arrays in NPZ files
with Unicode column names (`allow_pickle=False`), spatial profiles, validation
records, input hashes and figures. Raw restarts, binary dumps and full logs
remain external. To validate the package and regenerate all four figures:

```sh
python scripts/validate_package.py
python scripts/render_package.py
```

These commands require NumPy and Matplotlib but no Aurora access, raw dumps
or hardcoded workspace path. The manifest hashes the saved figures; rendering
with a different Matplotlib version can legitimately change image bytes.
Validate the original package before regenerating figures.

The standalone [checkpoint validator](../check_minkowski_checkpoint.py) finds
`tst/regression` in a checkout or accepts `ATHENA_REGRESSION_PATH`. It uses
header-only cohort selection and fully checks every selected rank's payload,
matching metadata, lapse, chi and all Sylvester positive-definiteness minors,
including ghosts. Selection equivalence passed on five one-rank fixtures and
one eight-rank fixture; a finite metric with positive determinant but negative
principal minors is still rejected. Sparse future profiling selects every
5000 M and the latest cohort, rather than parsing all 1000 M dumps. The
[collector](../collect_profiles.py) takes explicit `--runs-root` and `--output`
paths and does not submit, restart or alter jobs. Its portability smoke test
validated and profiled two real CPU checkpoint cohorts. These helper tests
do not constitute new GPU evolution results.

Reporting thresholds suppress meaningless growth fits but do not alter
plotted values or evolution. Gamma residuals are evolved Gamma fields,
not Gamma-minus-metric constraints. Coordinate-Euclidean/Frobenius vector
and tensor norms are labeled separately from physical metric norms. None of
these controls establishes strong-field, matter, AMR, reflection-coefficient,
full-spectrum or continuum stability.
