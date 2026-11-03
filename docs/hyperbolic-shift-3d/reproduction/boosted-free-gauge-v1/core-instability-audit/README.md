# Near-puncture failure: reconstructed metric stage

The immediate BG1 failure mechanism is now reproduced, rather than inferred
only from the location of the failed cells. This is a read-only diagnostic of
job9034868. No failed state is admitted, restarted or used as a physical endpoint.
Run `python3 docs/hyperbolic-shift-3d/reproduction/boosted-free-gauge-v1/audit_metric_stage.py`
from the repository. Input hashes, all four cell results, main-branch source
hashes and the analysis-script hash are recorded in `metric-stage.json`.

## What is isolated

At the last accepted time4.23M, all sampled donor metrics are positive definite.
For the four failed cells, reconstructing the actual FD4 metric RHS from that
state, including biased shift advection, centered shift gradients, curvature
and the tapered energy-form KO8 operator, gives

    g_stage = g_base + (dt/2) RHS(g_base), dt = .0005.

This matches every metric component of the failed capture in all four cells
within8.89e-16. Thus the captured metric identifies the first RK half-stage
(second RHS evaluation), at coordinate t1.41025 or4.23075M. This stage time is
inferred from reconstruction; it was not explicitly recorded by the worker.
The pre-step weakest eigenvalue is+.0557725859 and the stage value is
-.00142789446. Chi and lapse are positive at these cells.

For a unit eigenvector z of the pre-step weakest metric mode, the projections
z^T RHS z at the first failed cell are:

| Contribution | Projected rate |
|---|---:|
| Biased FD4 shift advection | -378.782435 |
| Lie stretching and trace subtraction | -4.712803 |
| -2 bar-alpha A | -0.369441 |
| Tapered KO8 | +155.164394 |
| Total | -228.700286 |

High-order advection of the sharp metric feature dominates the loss of positive
definiteness. KO is opposing it. This is not a newly negative lapse/chi stop,
and no core interpolation donor is involved in this local stencil: Cartesian
block neighbors supply direct values. The rate reconstruction also checks the
advection sign, indexing across blocks, and the actual KO contribution at the
failure, without proving that all earlier updates were correct.

Counterfactuals on the **same saved state**, with every other term fixed:

| Change to this first stage only | Minimum metric eigenvalue |
|---|---:|
| Actual FD4, full dt=.0005 | -.00142789446 |
| Actual FD4, full dt=.00025 | +.02717878010 |
| First-order upwind transport, full dt=.0005 | +.13149659327 |
| Main NGHOST4 biased-sixth transport, full dt=.0005 | -.07615139313 |
| Centered-fourth transport, full dt=.0005 | -.20636763405 |

These are diagnostic substitutions, not completed evolutions or recommendations
to lower the global method order. First-order upwind suppresses this stage's
overshoot, whereas merely raising transport order does not. Halving dt avoids
this particular stage failure; later stages/steps and the upstream evolution
are untested. All stencil donor metrics are positive before the step (smallest
eigenvalue .0557726); positivity of donors does not make the high-order update
positivity preserving.

## Stitching versus physical core resolution

The tracker is at radius.02318, the failed cells at.01866, and the largest radius
of the entire local derivative/KO stencil is.062385. The inner gauge is exactly
on its plateau out to.2; the restoration collar begins at.6. Neither the puncture
nor this stencil reaches either transition. Direct movement into the stitching
region is ruled out. Signals generated elsewhere are not ruled out; the earlier
large pole/clock residuals remain relevant evidence.

On the initial inner plateau the coordinate map is x=k X, t=T, with
k=.306142283363079. Thus there is no rapidly varying compactification Jacobian
near the failure. However the physical Cartesian resolution is coarse:

| N | Compact spacing | Initial lab spacing / M | Initial isotropic horizon-radius scale / spacing |
|---|---:|---:|---:|
| 16 | .01125 | .110243 | 4.535 |
| 20 | .00900 | .088194 | 5.669 |
| 24 | .00750 | .073495 | 6.803 |

The last column uses the reference isotropic radius M/2; it is a scale estimate,
not the measured late moving horizon size, and the boost mildly contracts the
initial x extent. Refining three to six outer shells did not refine this core.
The narrow pre-step eigenvalue trough spans only a few core samples, as shown
without smoothing in `metric-stage.png`. Core under-resolution is a plausible
upstream explanation, but a finer-grid solution is required to establish it.

## Comparison with main-branch Z4c

The comparison uses main commit c8d5c33e39089f005e416bf839652f6796fa6e82,
not the modified Z4c files on this experimental branch. The checked boosted
input is an example with v=.8660254 and M=1, not an identified matched v=.1 run.
Its parameter settings cannot be treated as the measured configuration of every
successful main-branch evolution.

| Item | BG1 | Main / checked boosted example |
|---|---|---|
| Exact inner boosted geometry | Lorentz-transformed Schwarzschild | Same construction; prior metric/K/lapse audit agrees to roundoff |
| Initial shift | Zero inside r=.2 by explicit user choice | Boost-coordinate shift |
| Inner lapse | Advective 1+log term plus telegrapher B divergence/evolution | Advective 1+log, with slow-start lapse enabled in example |
| Shift driver | Advective Lambda driver, eta=2/M | Advective Gamma driver; coordinate normalization must be matched for a controlled comparison |
| Core spatial operators | Centered/biased FD4, KO8 coefficient.5 with taper | NGHOST4 FD6/biased6, KO8 coefficient.1 in example |
| Resolution | Uniform core, initial delta X/M=.11024 | AMR; example base .25M and potential level3 spacing .03125M |
| Inverse-chi protection | Raw positive chi in geometric divisions; separate state floor; lapse-B guard only | Example chi_div_floor=1e-5 for geometric denominators |
| Constraint damping | Inner kappa1=.01/M | Example kappa1=.02/M |
| Algebraic projection in vacuum RK4 | After full step | After full step; not an every-stage safeguard |

These differences prevent attributing main's success to a single stencil or
boundary condition. In particular our inner gauge is not simply main Z4c
written in compact coordinates. The BG1 zero-shift choice and telegrapher lapse
can produce a different transient near the puncture. The general denominator
guard also differs, although the minimum core chi in the final pre-step snapshot
is about6.57e-4, so a1e-5 guard would not activate at that snapshot. Earlier floor
events and their downstream influence have not been isolated by this audit.

## Remaining causal test

The evidence establishes a local high-order advection overshoot, but does not
uniquely distinguish coarse spatial resolution, timestep sensitivity and a
sharp feature driven by the inner gauge. The clean next experiment is a bounded,
fresh-data time/space control through just beyond4.23M: first N16 with half dt,
then core refinement at a common sufficiently small dt, retaining the same
effective boundary penalty. Simply halving dt with current automatic scri
clipping changes that penalty and would confound the comparison. If the feature
persists under refinement, isolate the inner lapse driver next with a separate
main-like gauge control. Do not change resolution, gauge and dissipation together.
No new evolution or automatic higher-resolution campaign was submitted here;
the original N16-success gate and failed-run disposition remain unchanged.

`analyze_failure.py` also had an ancillary D-field label corrected: D is stored
in fields25:28, not28:31 (the inactive adaptive source). Regenerated localization
now reports the actual nonzero D values. The metric localization, failure time,
constraint histories, and existing failure conclusions were unaffected.
