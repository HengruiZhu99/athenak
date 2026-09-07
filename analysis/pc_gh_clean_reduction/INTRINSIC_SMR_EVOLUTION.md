# Repeated-stage SMR evolution: convergence screen fails

Eight serial runs using the exact f8d91ed4 production tree and saved executable
`intrinsic-point-restrict-001/source/athena-serial` completed in 110.583 seconds
summed wall time. Each evolves the independent sinusoidal off-constraint initial
state to t=0.02, with static seven-leaf periodic geometry, block sizes 8/16/32,
FD6, SSPRK3, KO0.3, eta2, kappa1, lapse-scaled reduction rate1 and point6_2d
restriction. `coherent_transfer=none` and `residual_shifted` are the matched arms.
All use dt=0.000125 (160 steps, 480 RK-stage exchanges); both finest cases also
use dt/2 (320 steps). No physical Einstein-data or black-hole run was performed.
This is evolution of a nonstationary PDE fixture, not a perturbation of an exact
stationary solution.

All runs finish with positive recorded health margins. Across their health
records, min w=0.999, rho=0.99902924, alpha=0.99803738 and minimum metric
eigenvalue=0.99609990. All89 component diagnostics are retained at t=0,.005,.01,
.015,.02, along with initial/final float64 restarts. An independent leaf-stencil
calculation verifies all78 reduction/intrinsic-curl/Q-curl RMS and maxima in each
final CSV to maximum normalized discrepancy 8.54e-15. The CSV uses volume2.21
(inactive-axis extent1.7); the comparison uses area1.3. Their RMS values agree
because that inactive extent is constant across all leaves.

## Signed convergence failure

Comparison uses the same physical leaves and coarse cell centers. Degree-five
local tensor interpolation, shifted at source-block boundaries, compares n8
with n16 and n16 with n32, then maps the second signed difference to n8.
A smooth interpolation control converges above order6.2. The frozen exploratory
screen requires group order>=3, alignment>=.99 and independent temporal
contamination<=.05. Both arms fail:

| Six-point comparison | Group | Norm-ratio order | Signed alignment | Temporal fraction |
|---|---|---:|---:|---:|
| Ordinary transfer | first20 | 3.946 | 0.00719 | 5.70e-5 |
| Ordinary transfer | all50 | 4.453 | 0.06524 | 5.00e-5 |
| Residual reconstruction | first20 | 5.238 | -0.47534 | 2.91e-6 |
| Residual reconstruction | all50 | 4.332 | -0.22875 | 3.50e-6 |

These norm ratios are **not valid Richardson orders** because alignment fails.
All50 individual rates/alignment/temporal ratios and signed arrays are preserved.
Temporal fraction compares the finest dt/2 state difference to the n16/n32
spatial difference with coordinate-area normalization; reducing dt does not
resolve the alignment failure.

A separately recorded interpolation sensitivity control uses ten active source
points (degree nine) and removes a constant anchor during interpolation to
reduce background cancellation. Its smooth interpolation control has rates
10.30/10.50. The initial analytic interpolation difference is only 3.71e-5 of
the evolved first20 difference for ordinary transfer and 1.43e-6 for residual
transfer. Nonetheless all50 alignment remains negative (-0.237 and -0.486),
and its norm ratios change to 1.783 and 0.831. The evolved comparison is sensitive
to interpolation; neither result establishes an asymptotic regime. This control
does not prove that interpolation error on the evolved interface structure is
negligible. The original six-point failure remains unchanged in the ledger.

## What the diagnostic histories do and do not show

At n32, ordinary-transfer reduction RMS changes from .01951195 to .01912559;
intrinsic curl RMS from .009500575 to .009312451; Q-curl from .01654573 to
.01621811. The residual arm has very similar total RMS values. These totals
include the intentionally nonzero analytic constraints and can hide small
interface errors. H changes from .28922576 to .29039450, which is not a vacuum
constraint-convergence claim. No blow-up appears in this short interval, but
survival and decreasing total reductions do not overcome the field-alignment
failure or establish a benefit from reconstruction.

Next: synchronized per-operation budgets during evolution, including the
source residual, ordinary transfer and reconstructed ghosts, to distinguish
interface-local differences from propagated bulk differences before changing
another operator or extending the run. The current experiment does not qualify
repeated injection, 3D/CUDA interfaces, physical smooth data, punctures or binary
merger. Production defaults and equations were not changed this turn.

## Reproduction

The frozen PLAN.md/gates.json, inputs, exact executable/input hashes, run times,
component CSVs, health summaries and analysis results are indexed under
`qualification-runs-20260907/pcgh-clean-reduction/intrinsic-smr-evolution-001/`.
Raw restarts and signed NPZ arrays remain under the same named directory in
`/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/` with a hash inventory.

Use `run_intrinsic_smr_evolution.py --binary <saved executable> --fixtures
<intrinsic-point-restrict-001> --output <new directory>`. Analyze with
`analyze_intrinsic_smr_evolution.py --runs <directory> --output <new analysis>`;
repeat with `--points 10` for the sensitivity control. The diagnostic,
interpolation-sensitivity and health-summary scripts accept `--runs` and a new
`--output` JSON file. All analysis checks used the external venv Python with
`-W error`. No remote resources were used.
