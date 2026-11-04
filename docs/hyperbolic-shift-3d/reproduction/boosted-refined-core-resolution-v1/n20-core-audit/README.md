# N20 puncture robustness audit against telegrapher_lapse

This is a read-only forensic analysis of the last accepted state and rejected
stage of job9037163. The checkpoint remains unadmitted: it is neither restarted
nor used as a successful physical endpoint. No campaign settings, jobs or frozen
binaries were changed. Reproduce with `../audit_n20_metric.py` from any directory.
Input hashes, frozen source hashes, reference source hashes and script hash are
in `audit.json`. The remote reference branch was checked and resolves to
`origin/project/telegrapher_lapse` commit
`b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a`.

## What is and is not an accuracy failure

Large unexcised constraint norms near a puncture are not evidence, by themselves,
of inaccurate exterior evolution. The singular puncture and its interior should
not be used to demand smooth fourth-order convergence. No fixed-domain exterior
convergence failure has been established by N16/N20 moving-cut histories.

The distinct robustness failure is loss of positive definiteness of the evolved
conformal metric. Even inside a horizon, the present non-excised evolution must
evaluate inverse metrics there. It cannot proceed through an indefinite metric.
This is a failure of this discrete evolution, not a claim that an interior
constraint spike is an observable physical error.

## Resolved samples show a feature only a few cells wide

At the last accepted time6.2082M the failing cell is at
(-.01575,.00225,.00225), approximately2.01 grid spacings from the tracker.
The core spacing is.0045, equivalent initially to lab spacing.04410M because
x=kX with k=.306142283363079 and t=T on the inner plateau. The mapping is a
constant scaling there, not a rapidly varying compactification Jacobian.

The conformal metric eigenvalues at that cell are
[.00529370,12.81485,14.74102], condition number2784.64, with determinant one
up to rounding. The rejected stage has minimum eigenvalue-.00278346. Along
the transverse y line, consecutive pre-step minimum eigenvalues are

    offset/h:       -3       -2       -1        0        +1       +2       +3
    lambda_min:   .73470   .31600   .08911   .005294   .30444   .65541   .77323

Thus the deepest metric trough is represented by one cell and its shoulders by
only a few cells. The evolved connection magnitude is418.88 at this cell,
versus36.57 and60.07 at its immediate y neighbors. Shift components oscillate
on adjacent cells; `axis-samples.csv` preserves them without smoothing.
This is strong evidence of grid-scale structure and poor local resolution,
not a measurement of the continuum feature width or proof that more cells alone
will fix it. N20 already failed despite N16 surviving.

All axial derivative/KO donor points used in the local audit have r<=.03390,
well within the inner gauge plateau r<=.2 and far from restoration beginning
at.6. Direct entry into stitching is excluded. Propagated effects from elsewhere
are not excluded. No time-resolved interior spatial dumps exist around4M.

`core-gradients.png` shows the three coordinate lines through the failing cell,
using the last accepted state, not an indefinite-state constraint calculation.
The figure was visually inspected after generation.

## Concrete differences from the Cartesian reference

Reference input settings are taken from
`experiments/telegrapher_aurora_20260929/inputs_v4/g1.5_bhahaha_L5.athinput`.
That example has boost.745356 and AMR; it is not a matched v=.1 control. The
branch implementation, rather than an assumed generic Z4c scheme, is compared.

| Item | BC2 hyperboloidal core | telegrapher_lapse reference |
|---|---|---|
| Geometry | Exact boosted-hole initial geometry, native compact map | Exact boosted-hole Cartesian initial geometry |
| Initial shift | Explicitly chosen zero inner shift | Boost-coordinate shift |
| Inner lapse | Advective1+log plus telegrapher B | Same type of gauge, no slow-start in checked input |
| Mass-scaled gauge/damping parameters | tau=.1M, kappa=.2M, eta=2/M, kappa1=.01/M | Same values in M=1 units |
| Spatial derivatives | Centered fourth order, biased fourth transport | Centered sixth order, biased sixth transport |
| KO | Coefficient.5, tapered block-local negative-square KO8 | Coefficient.1, conventional KO8 using ghost cells across blocks |
| Small chi | Direct state floor1e-8, origin-centered positive-floor patch; raw positive chi in geometric divisions | chi_div_floor1e-5 in geometric and gauge coefficients, evolved-chi flooring disabled by default in checked input |
| Vacuum RK4 algebraic projection | Full-step projection, positive-definiteness checked | Full-step projection too; nonpositive determinant is replaced by1 for projection scaling, which does not repair an indefinite metric |
| Resolution | Uniform core, lab spacing.04410M at N20 initially | Tracker AMR, nominal finest.03125M in checked level5 input |

The reference inverse-chi guard does not explain the immediate failing cell:
its pre-step native chi is.00044527, above the corresponding guarded range.
One of2942 nominal core points within.04 of the tracker is below the mapped
AF threshold in this snapshot; historical effects remain possible. In constant
inner coordinates, chi_AF=Omega^2 chi_native/k^2. A mapped AF denominator
floor therefore cannot be compared to native chi by copying the same number.
The hyperboloidal telegrapher coefficient guard is separate from geometric
denominator protection and also requires this coordinate normalization if an
exact AF gauge comparison is intended.

The geometric kernels also have different off-constraint implementations:
the compact kernel constructs Penrose metric curvature and explicit Z terms;
the Cartesian kernel constructs its conformal Ricci expression using evolved
Gamma derivatives. The old inner-gauge experiment explicitly did not replace
the entire geometric RHS with the Cartesian Z4c kernel. This audit does not
prove an additional algebraic error in those transformed equations.

## Important coordinate normalization of the Gamma driver

To reproduce the *same physical gauge* under the constant spatial map x=kX,
with unchanged time t=T, start from the Cartesian determinant-one conformal
metric. The physical spatial metric acquires k^-2 and its determinant acquires
k^-6, so its determinant-one conformal metric has unchanged components:

    gtilde_x = gtilde_X, chi_x = k^2 chi_X,
    Lambda_x = Lambda_X/k, beta_x = k beta_X.

Consequently a Cartesian driver

    (partial_t-beta_X.grad_X) beta_X = Gamma_X - eta beta_X

transforms to

    (partial_t-beta_x.grad_x) beta_x = k^2 Lambda_x - eta beta_x.

The implemented inner coefficient is1, not k^2=.09372310. Relative to this
coordinate-transformed reference the connection driving coefficient is10.6697
times larger. Using coefficient1 is a legitimate different coordinate gauge,
but it is not the exact pullback of the reference gauge. Matching the formal
equation names and mass-scaled eta alone does not match the gauge. This is a
particularly useful difference to isolate given the narrow Lambda spike.
It is an analytic coordinate comparison, not proof of the cause of failure.
No coefficient was changed in this investigation.

D is a separate outer auxiliary. Its large failed-stage values must not be
mistaken for the inner shift driver: on the inner plateau beta couples to
Lambda, while D is decoupled from beta there.

## Dissipation and local stage sensitivity

The failing cell has local block indices(16,0,0): it lies in the first cell on
two Cartesian block faces, y=0 and z=0. Geometric/transport derivatives cross
those faces, but the block-local KO operator does not. Only4/1/1 complete
five-point difference lines contribute in x/y/z, compared with5 per direction
away from block faces. Negative energy of KO does not imply pointwise metric
positive-definiteness preservation.

Using the pre-step state and the implemented FD4/KO metric formulas, the
weakest-mode projected rate contributions are:

| Contribution | Projected rate |
|---|---:|
| Curvature | +.00709 |
| Biased advection | -5.14352 |
| Lie stretching | -.00818 |
| Block-local KO8 | -35.29888 |

Here KO pushes that weak mode downward; unlike the earlier coarse BG1 audit,
it does not oppose the local collapse. The resulting first half-stage estimate
has positive eigenminimum.00123977. It does **not** reproduce the captured
rejected metric (component difference.1674), so we do not identify the actual
failed RK stage or assign these percentages to its cause. The full intervening
stage states/RHSs were not saved. The prior exact first-half-stage BG1 diagnosis
must not be copied to N20.

At this one saved state, changing only transport to the reference sixth-order
stencil gives first-half-stage eigenminimum.00096053; adding cross-block KO
at the same coefficient gives.00161415; half dt gives.00326913. These are
algebraic local sensitivity estimates, not evolved counterfactuals or successful
repairs. Simply raising transport order is not supported as a positivity fix.

## Most discriminating follow-up

The evidence supports a grid-scale, ill-conditioned inner metric/connection
feature interacting with a non-positivity-preserving discretization. It does
not establish a unique upstream cause or exterior loss of convergence.

The first matched-gauge control should isolate the Gamma-driver spatial
normalization, preserving the effective scri penalty, initial data, dt and
numerical operators. A separate control should compare denominator protection
and core KO across block faces. Do not change all of them at once. Any such
control should record local minimum eigenvalues, connection/shift gradients,
instantaneous boundary residuals and same-domain exterior norms, with bounded
checkpoint cadence near the instability. No new evolution was launched here.
