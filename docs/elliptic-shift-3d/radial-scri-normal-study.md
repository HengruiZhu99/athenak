# Radial scri jet study, 2026-10-27

This is a frozen-state **diagnostic**, with no elliptic solve, time advance, field
projection, gauge change, or production engine modification. It identifies a real
initial normal-derivative truncation error and rejects the inference that removing
that error alone establishes evolved scri compatibility. Every evolved candidate
still fails the unchanged maximum pole-numerator target `1e-7`.

## Inputs and scope

The old 47,088-cell `amr-log-halo-coupled-20261027/coupled.bin` and the genuinely
evolved Hermite version16
`hermite-supported-coupled-20261027/coupled.checkpoint` both represent the midpoint
`t=0.0001`. The latter is **not** its final `t=0.0002` state: its recomputed midpoint
maximum is `5.9886790127e-7`, whereas the parent's actual completed evolution reports
`1.0845869545e-6` at the final time. These times/results must not be conflated.

Both use N6, the seven radii `.45,.55,.60,.65,.70,.75,1`, the same original
Minkowski-reference gauge, physical independent Lambda, direct signed fields, log
positive storage, and the actual Omega=0 boundary. Original checkpoint and initial
sample hashes are in the analysis manifest. The exact initial source is the frozen
`amr-core-pilot-20261027/initial.dat`, evaluated from the independent isotropic
M=1/3 spherical initial-data routine. All 216 boundary points are retained.

The new host-only prototype headers are
`src/mesh/cubed_sphere/radial_scri_jet_probe.hpp` and
`anchored_radial_scri_probe.hpp`. The existing sampler, probes, engine, checkpoint,
native owner, and PDE headers were left unchanged. The prototype follows actual
aligned active angular rays through any configured number of shell layers and
face subdivisions. It uses physical radii from the map and actual active IDs, not
shell ghost values. Changed angular lattices/shell AMR are unsupported. Insufficient
points cause an explicit error; the requested degree/count is never reduced.

## Free coordinate bases and full chain rule

The interpolation spaces are complete polynomials in one of

- `r-R`,
- `(R^2-r^2)/(2R)`, or
- `t=-log(r/R)`.

The constant, linear, quadratic, and every other coefficient remain free. The
actual stored boundary value is the anchor. Anchored differences preserve constant
fields exactly without subtracting a known physical profile or RHS. A polynomial
in r exactly reproduces physical r-polynomials; a polynomial in t exactly reproduces
t-polynomials. Physical r-linear/quadratic fields are generally only approximated
by the log basis; the AD tests measure that truncation rather than calling it exact.

For a fit `p(t)`, the implemented physical derivatives are

```
f_r  = -p_t/r
f_rr = (p_tt+p_t)/r^2.
```

The general code uses `f_r=zeta_r*p_zeta` and
`f_rr=zeta_r^2*p_zetazeta+zeta_rr*p_zeta`, including the nonzero second term for
both Omega and log coordinates. A separate analytic chain test uses R=1.7 and
arbitrary positive radii. The Hermite pullback keeps the existing full Cartesian
J/H chain, `q_0= r_s G`, `q_00=r_s^2 H` for the current affine-r maps, and
`q_0A=r_s D_A G+(D_A r_s)G`. It does not treat patch q0 as a scalar field.

The second alternative is an overdetermined degree 6 fit to the actual trace and
8,12,or14 active samples, with all six nonconstant coefficients free. It uses
reorthogonalized QR in scaled coordinate space. This is a field-independent
finite-difference/least-squares derivative operator, not a projection of evolved
fields. It is retained as a **failed** alternative below.

All plans predeclare condition `<1e8`, polynomial reproduction `<1e-10`, and first
and second derivative L1 amplification no more than four times the original
last-layer seven-node physical-r plan. Reproduction is checked using the stored
Real weights, after extended-precision coefficient construction. No tolerance
was changed after seeing the evolved residuals. For perturbations bounded by eps,
`|delta D F| <= eps*sum(abs(weights))`; a sign-aligned perturbation verifies the
first-derivative worst-case bound. This is a data-noise bound, not a scientific
acceptance criterion.

## Initial truncation and evolved counterexamples

The independent exact outer initial branch has, for `t=-log r`,
`log chi=-(M/2)t^3+O(t^4)` and
`log(alpha/Aref)=-(3M/4)t^3+O(t^4)`. Thus the exact first and second radial
derivatives vanish at scri for **this initial slice**. They are only comparison
oracles; no zero coefficients or radial symmetry are imposed on the fitted fields.

| Method | Exact initial max pole | v16 midpoint max pole | v16 midpoint max Axx |
|---|---:|---:|---:|
| r polynomial,7nodes | 7.5989520e-7 | 5.9886790e-7 | 5.9877059e-7 |
| log polynomial,7nodes | 6.6605699e-10 | 5.3811764e-7 | 1.9455422e-7 |
| r polynomial,9nodes | 8.9923617e-8 | 2.9441934e-6 | 2.9441934e-6 |
| log polynomial,9nodes | 3.9137582e-12 | 1.1218766e-6 | 1.1218766e-6 |
| log degree 6 LS,13nodes | 6.1039600e-9 | 1.2568550e-6 | 1.0329014e-6 |
| log degree 6 LS,15nodes | 4.2815456e-8 | 1.2302487e-3 | 1.6398619e-4 |

Seven log nodes reduce the initial chi first-normal error from `-6.1877181e-7`
to approximately `-5.42358e-10`, with derivative L1 ratios `0.86012` and `0.77329`.
The Omega basis is worse: its initial pole is `1.8781297e-5`. The nine-node
physical-r initial result passes the pole target but the evolved result worsens.
The 13-node log LS plan has smaller L1 ratios `0.69015/0.40255`, yet worsens the
evolved result. The 15-node collar reaches into the transition region and gives a
large evolved defect despite acceptable linear conditioning. These are direct
counterexamples to choosing a stencil only by initial radial accuracy or L1 size.
No claim that all remaining discrepancy is a derivative error is warranted.

Ten-node r interpolation is rejected before field evaluation because its L1
ratios `13.5045/23.3370` exceed 4. Eleven-node r interpolation is rejected at
condition `2.0907191e9`. Both old and v16 rejection records are retained. There is
no retry with relaxed guards or lower-degree fallback.

At the log7 v16 maximum, LambdaX is `5.3811764094e-7`, at row35,
`x=(.6469966392,.5391638660,.5391638660)`. Its exact additive decomposition is

| Contribution | Numerator |
|---|---:|
| Metric normal first jets | +1.0396171450e-6 |
| Metric angular first jets | -5.6133937865e-7 |
| Independent Lambda | -4.4424568144e-9 |
| Trace/Theta normal first jets | +1.0277065114e-7 |
| Trace/Theta angular first jets | +1.1073583004e-9 |
| Curvature contraction with grad Omega | -3.9595677993e-8 |

The sum agrees to `2.12e-22`. Axx is likewise split into its metric-value baseline,
chi normal/angular, metric normal/angular, independent Lambda, and curvature
terms. The maximum decomposition error over all candidates/rows is below
`5e-15`. These are algebraic diagnostics on immutable input data, not modifications
to an evolution. The comparable metric-normal/angular terms rule out identifying
the entire remaining problem with the initial chi normal truncation, or fixing it
by replacing the independent Lambda with the metric connection.

## Independent accuracy and complete Minkowski controls

Twenty related geometry cases cover 1/2/7 shell layers, 6/7 original radial points,
three affordable N per family, and face subdivision2. Eight independent analytic
Cartesian fields include a constant, angular dependence, radial-angular products,
a Cartesian polynomial, `(r-1)g(n)`, `(r-1)^2g(n)`, a Cartesian linear field, and r.
All actual nodes, two-cell edge/corner bands, and exact face seams/corners are
included. All trace/constant/finite/shape tests pass. Normal-only second-derivative
orders are approximately 3.95–4.02 for 6 nodes and4.98–5.19 for 7 nodes. Seven additional
cross-layer LS geometry cases are recorded without claiming that their varying
physical support defines a uniform-grid convergence family.

The full Cartesian angular+radial first-derivative errors have coarse observed
rates below 4, inherited from the angular reconstruction regime. These data do
**not** establish global fourth-order accuracy. They do establish the intended
normal-derivative consistency independently of the initial black-hole profile.

Four complete local Minkowski controls (r7,log7,log9,log-LS13) give exactly zero
stored constant jets; full conformal tensor pole, regular scri limit, and lapse
clock residuals remain at roundoff. No native or research evolution was run with
the new radial plans. The complete frozen wormhole regular-limit RHS is not
claimed because fresh globally solved beta derivative jets were not installed.

## Reproduction and recommendation

From the isolated repository:

```
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/run_radial_scri_jet_probe.py \
  --output docs/elliptic-shift-3d/evidence/NEW-radial-study --matrix all
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/run_radial_scri_minkowski.py \
  --output docs/elliptic-shift-3d/evidence/NEW-radial-minkowski
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/analyze_radial_scri_jet_probe.py \
  --input docs/elliptic-shift-3d/evidence/radial-scri-jet-final-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-radial-analysis
```

Each subprocess has a 180-second bound and its own owned process group; builds
use private frozen sources without CMake/shared-binary edits. Final capsules,
exact commands, failure stderr, rowwise lossless CSV.gz, input hashes, runtime
hashes, and source-dependency proof are under:

- `evidence/radial-scri-jet-final-20261027/`;
- `evidence/radial-scri-minkowski-final-20261027/`;
- `evidence/radial-scri-jet-analysis-20261027/`.

The first Minkowski runner attempt failed **before compilation** because the
local Python does not support `TarFile.extractall(filter=...)`. That failure is
preserved in `radial-scri-minkowski-20261027/failure.json`; explicit checked member
extraction then succeeded in distinct retry/final directories. Earlier radial
attempts are preserved separately. No process/permission denial is outstanding.

A seven-node log-coordinate normal operator is a reasonable **optional numerical
experiment**, since its formal accuracy, initial truncation, and amplification
controls pass. It is not a demonstrated solution of the evolved scri problem.
If integrated later, it needs an explicit checkpoint-bound option and a fresh
bounded coupled evolution from initialization: an offline stencil swap cannot
validate its effect on accumulated stage errors or regular-limit closure. Wider
interpolating or LS collars should not be adopted on the present evidence.
The next scientific gate remains the original full tensor/gauge boundary residual,
including the metric-normal/angular compatibility seen above; it must not be
weakened or replaced with the normal-fit residual.
