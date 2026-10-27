# Discrete connection consistency audit

This is a failed gauge consistency qualification, not a new evolution or an
adopted elliptic operator. All cells and all scri nodes are retained. The input is
**the actual final accepted state** at t=0.0002 of the failed optional-affine N6
seven-shell wormhole pilot, not its intermediate warm-start checkpoint.

The source is `affine-coupled-n6-accepted-20261027` (process exit **1**, scientific
scri failure); accepted binary SHA256
`a12a6bb083a7371e8f1f0a4f2d306fe6f89e1819739a50d139664c2d4f7a80b6`.
The snapshot SHA256 is
`0ca8a2bd401a109987e6fa019449d321fc060205bb5836e70f4173976ea1101b`.
The immutable first attempt is `evidence/discrete-connection-first-20261027`.
The qualified fresh old-operator comparison and full arrays are in
`evidence/discrete-connection-baseline-20261027`; its private source archive SHA256
is `667590aece90782a6c47a8e11a9d29adc8ee74828efd483a94357836c071b5ba`.
Compiler dependency inspection reports **zero live project dependencies**.
`evidence/discrete-connection-regions-20261027/summary.json` contains every region,
physical and conformal norms, and top ten points for every term.

## What is differentiated

Let `D` denote the actual first derivative of the stored metric, including global
volume halos, Cartesian/shell and shell/shell transfer, scri interpolation and the
selected Hermite surface sampler. Define the measured discrete connection by

```
V_D(g) = inverse(g) contracted with Christoffel(g,Dg).
C = -2 alpha A
H = actual metric RHS - C
actual metric RHS = C + H.
```
The actual full tensor metric RHS is sampled at every active point; the surface
metric rate is the **actual** `BoundaryRates` result. Beta is total beta: analytic
Minkowski reference `-x` plus the solved delta beta. In the optional-affine case,
body and surface delta-beta jets use analytic affine plus the sampled remainder.
No metric, Lambda or curvature constraint projection is performed. The metric-rate
value identity `C+ShiftMetricRate.h = actual metric RHS` holds within 2.665e-15.

The independently checked target is

```
d/d epsilon V_D(g + epsilon actualMetricRHS) at epsilon=0
= MetricConnectionVariation(g,Dg, actualMetricRHS,D(actualMetricRHS)).
```
The finite perturbation changes all volume and surface metric values and rebuilds
the original halos and surface jets. The inverse and Christoffel connection are
recomputed for both signs. Other evolved fields are fixed. This differs from the
expanded continuum vector equation's residual

```
expanded = DV_g[ShiftMetricRate(beta) + C] = L_g beta - S
```
when direct second derivatives, product rules and chart reconstructions do not
commute with `D`. Eta is zero in this selected configuration.

## Same-geometry old-operator control

The old operator control keeps **all accepted evolved fields and all scri fields
bitwise identical**, changes only the optional affine derivative flag to OFF,
and freshly solves the original old expanded shift equation. It is not a
separate evolved trajectory. The same warm-start is allowed; solve cost and
residual are reported, not presumed. It converged in 16 iterations / 1.60 seconds,
physical original RMS 5.384e-11 and point maximum 1.549e-9. Each audit completed in
under five seconds; all process limits are 180 seconds.

| All-cell vector norm | Accepted affine beta | Fresh old-operator beta |
|---|---:|---:|
| Expanded original physical RMS | 6.1335e-11 | 5.3840e-11 |
| Actual discrete V_t physical RMS | 2.20200 | 2.20317 |
| Actual discrete V_t conformal RMS | 2.84287 | 2.84338 |
| D2 beta versus D1(D1 beta), physical RMS | 2.31192 | 2.31305 |
| Product Leibniz error, physical RMS | 0.147876 | 0.147876 |
| Remaining shift metric products/halos, physical RMS | 0.046260 | see manifest |
| Actual discrete surface V_t, uniform-point vector RMS | 0.135953 | 0.00177780 |

The bulk mismatch is therefore inherited by the old expanded operator. The affine
option separately amplifies the surface discrete connection mismatch on this
geometry. The tensor scri pole remains 8.925092e-5 for this same accepted geometry
in both recomputations. It is not evidence that the old trajectory has that pole,
or that a shift change alone can repair the fixed state's scri compatibility.
The recorded pole is a maximum over surface nodes of this RHS evaluation; it is
**not** a maximum over prior RK stages.

Physical weights are the positive map cell volumes times `sqrt(det(g))/chi^(3/2)`,
normalized by their sum, evaluated with the solver's log-weight contract. Conformal
weights are map cell volume times `sqrt(det(g))`. No extra Omega factor, floor,
point exclusion or exterior budget is used. Regional RMS values renormalize by
the full region's weight. Scri uses uniform-point vector RMS and point maxima;
there is no finite physical spatial volume interpretation at Omega=0.

| Region | Count | Affine actual V_t physical/conformal RMS | Old actual V_t physical/conformal RMS |
|---|---:|---:|---:|
| r < 0.1 | 56 | 0.0004150 / 0.0002685 | 0.0004181 / 0.0002728 |
| All Cartesian cells | 1728 | 0.12705 / 0.22330 | 0.13600 / 0.23844 |
| 0.62 <= r <= 0.70 | 2160 | 8.53046 / 8.61005 | 8.53069 / 8.61028 |
| r >= 0.8 | 1080 | 0.016802 / 0.016799 | 0.007676 / 0.007673 |

The top affine component mismatch is 15.22748 at active point 7905,
`x=(0.1101753818513,0.1101753818513,0.6610522911079)`, r=0.6791666667,
shell block36 in layer `.65..70`. The direct-D2/composed-D1 contribution there is
15.36262. The product-rule maximum 1.35219 occurs at r=0.6708333333 in the same
layer. This localizes the largest bulk failure near the steep lapse front,
rather than at the puncture. This localization does not remove any puncture cells
from acceptance.

## Complete decomposition and finite-epsilon certificate

`audit_discrete_connection.cpp` records the following separate contributions:

1. `D(-2 alpha A) + 2[(D alpha)A + alpha(D A)]`, the nodal physical product-rule
   error;
2. `-2[(D alpha)-alpha_jet.d] A`, the physical lapse reconstruction derivative
   difference, including analytic reference-lapse versus nodal differentiation;
3. `-2 alpha[(D A)-A_jet.d]`;
4. the difference between the actual differentiated shift metric rate and its
   expanded rate;
5. its direct-beta-D2 versus differentiated-beta-gradient part, retaining all
   curved metric coefficients;
6. remaining metric products and halo contributions.

The sum closes against actual-minus-expanded to physical RMS 1.792e-14 and
point maximum 2.558e-13 (surface maximum 1.388e-16). The non-product chart
contribution has physical RMS 0.0038941. It is separately visible, rather than
mislabelled as a pure product-rule error.

For the actual metric RHS direction, central finite perturbations at epsilon
1e-4 agree with the discrete Frechet prediction to physical RMS 3.459e-10 and
point maximum 3.186e-9; at scri the maximum is 3.411e-14. Epsilon values 1e-4,
1e-5, 1e-6 and 1e-7, full fields, and every error are retained. The finite-epsilon
check verifies the derivative of the implemented metric-built connection. It does
not make the expanded residual an adequate discrete-gauge certificate.

## Affine surface cancellation: independent complementary falsifier

The large affine surface error has a separate, directly verified mechanism.
`probe_affine_surface_trace.cpp` differentiates the same accepted delta-beta field
in two ways: full-trace Hermite sampling, and the actual affine decomposition
`D_sample(beta-A)+D_analytic(A)`. The metric and shift values are fixed. A second
control replaces **only the outer Dirichlet trace** by exactly zero while keeping
the nonzero volume field and its same bounded global affine fit. This is a
synthetic boundary consistency test, not a new physical evolution or a puncture
beta pin.

The final evidence is `evidence/affine-surface-zero-trace-20261027`, private source
archive SHA256
`2d1717a6870922411d569e3571393fa87112e093d709a551ad96150e6ba03a6d`.
All216 surface points of each case are retained; runtime2.30seconds. The initial
actual-trace-only probe remains frozen in `affine-surface-trace-20261027`.

| Quantity | Actual trace | Exactly zero outer trace |
|---|---:|---:|
| Maximum absolute delta-beta trace | 8.393e-9 | 0 |
| Maximum global affine slope | 0.0849433 | 0.0849433 |
| Tangential gradient difference | 0.00117483 | 0.00117483 |
| Coordinate-radial gradient difference | 4.97e-15 | 4.96e-15 |
| Cartesian Hessian difference | 0.0111243 | 0.0111243 |
| Full-trace tangential gradient | 1.053e-8 | 1.727e-18 |

Here tangent means the derivative-index projection `P=I-n n^T`, n=x at the unit
coordinate sphere; it is not substitution of a metric-normal component convention.
The measured first-derivative difference equals
`D_analytic(A)-D_sample(A)` and the second-derivative difference equals
`-DD_sample(A)` to at most1.471e-13 over both cases. Thus the cancellation artifact
is field-linear and independent of this tiny outer trace. Angular MLS reproduces
its chart polynomial basis, whereas a global Cartesian affine field restricted
to the sphere is generally not a chart polynomial. The pure-affine test with
zero remainder does not test this complementary cancellation.

A possible future discretization must differentiate the same imposed trace and
the coordinate-radial data consistently, rather than correcting the evolved
field. This audit does not adopt such a change. In particular, zero trace does
**not** require a zero Cartesian tangential-tangential Hessian. At the unit sphere,
for a scalar Cartesian component with traceF=0,

```
P grad f = 0
P HessCartesian(f) P = G P,  G = n dot grad f.
```

The raw full-trace tangential-tangential Hessian maximum0.00430928 in the zero-trace
case is therefore not itself a violation. The surface covariant Hessian subtracts
the radial-curvature term. Mixed normal-angular derivatives likewise retain
angular derivatives ofG and map terms. Silently zeroing Cartesian second
derivatives would change the equation and is not proposed here.

## Independent arbitrary perturbations and limitations

`test_discrete_metric_frechet.cpp` independently builds a smooth nondiagonal SPD
metric with determinant spanning approximately 1.43969..1.49183, and arbitrary
nonspherical, nontracefree symmetric metric-rate data. These are synthetic
consistency data, not wormhole initial data. The volume and boundary metric and
rate values are sampled independently; analytic Cartesian jets are generated by
second-order automatic differentiation. Central finite perturbations differentiate
the actual halo/surface interpolant, while a separate continuum comparison uses
analytic metric and rate jets. No shift solve or evolution occurs.

The first immutable family is `evidence/discrete-metric-frechet-20261027`, private
source archive SHA256
`839f586d2d34b419910c989fdff0fabb7fa82d6fcdbb0ba65168db00e4a0c37c`.
N6/N8/N10 use the same seven shells and eight static Cartesian blocks, at most
50,000 cells. A further N6 case refines one actual Cartesian block, retaining all
coarse/fine errors. At epsilon 1e-4, the finite-Frechet body RMS errors are
9.60e-11 / 1.10e-10 / 1.26e-10 and surface RMS errors are
1.29e-10 / 1.41e-10 / 1.71e-10. These small discrepancies verify the independently
measured variation, not the spatial accuracy of the interpolant.

The separate continuum body truncation RMS decreases
6.387e-4 -> 2.105e-4 -> 8.992e-5; the latter observed order is about3.81,
**below a strict fourth-order acceptance requirement**. Surface RMS decreases
9.297e-4 -> 2.377e-4 -> 9.833e-5. The refined N6 maximum continuum error is
0.005046, versus 0.001346 on uniform-core N6; coarse/fine transfer error remains.
All raw points and epsilon errors are retained. The second unrelated nonspherical rate direction is also complete. The full
8-case family, including both refined-core controls, is frozen in
`evidence/discrete-metric-frechet-two-directions-20261027`, source archive SHA256
`ced6269887c81448f254b33213fc13b67d476073832cbec87573ef9d1002af44`.
Its raw samples include both Frechet and continuum errors at every node and five
epsilon values 1e-3 through 1e-7. The second direction's bulk continuum RMS is
0.00130894 / 0.000470908 / 0.000203360, observed orders 3.55 and3.76; this also
fails a strict fourth-order qualification. At epsilon1e-4 its finite-Frechet bulk
RMS errors are 4.47e-11 / 6.04e-11 / 9.22e-11. First-direction results repeat
bitwise across the immutable one-direction and two-direction families.

The complete regional report and plots are
`evidence/discrete-metric-frechet-analysis-v2-20261027`:
`finite-frechet-error.png` separates finite-perturbation truncation/roundoff from
`continuum-consistency.png`, which plots the spatial error and an N^-4 guide.
Surface edges/corners and refined/coarse Cartesian cells are separate diagnostic
regions, with no acceptance exclusions. On refined N6, first-direction Frechet
RMS in all1728 refined cells is 2.006e-10, while continuum RMS is 0.000379525.
The 24 actual surface corners have first-direction N6 continuum RMS0.00152626
and finite-Frechet RMS1.281e-10. Thus small Frechet discrepancy must not be
confused with small interface or continuum error.

The initial analysis attempt failed because NumPy sanitized decimal epsilon
column names; its `failed-attempt.json` is retained in the earlier analysis
directory. The corrected reader preserves column names explicitly. No raw
samples or physical test data were changed.

A pure composed first-derivative operator is not adopted here. Centered first
derivatives have a Nyquist zero on a uniform periodic stencil, so composing them
can introduce a checkerboard nullspace that the direct second derivative does not
have. Physical boundaries and mapped interfaces can disturb this exact mode
without supplying a satisfactory coercivity bound. The parent's separate
conditioning/spectrum prototype must be qualified before a possible operator,
source and boundary redesign. No finite-difference tolerance is relaxed; no
continuum equation or selected gauge is changed by this diagnostic.

Reproduction commands are the complete commands in each manifest, or:

```
python3 tst/elliptic_shift_3d/run_discrete_connection_audit.py --output NEW_DIRECTORY
python3 tst/elliptic_shift_3d/analyze_discrete_connection_audit.py --input BASELINE_DIRECTORY --output NEW_DIRECTORY
python3 tst/elliptic_shift_3d/run_discrete_metric_frechet.py --output NEW_DIRECTORY
python3 tst/elliptic_shift_3d/analyze_discrete_metric_frechet.py --input FRECHET_DIRECTORY --output NEW_DIRECTORY
python3 tst/elliptic_shift_3d/run_affine_surface_trace.py --output NEW_DIRECTORY
```

The runner always creates a new evidence directory, privately unpacks the accepted
source capsule, hashes copied libraries, records compiler dependencies, and runs
one CPU worker with a 180-second process cap. No native AthenaK task integration,
MPI/GPU support, gauge success, or completion of the coupled scientific milestone
is claimed by these audits.
