# Shared compact slicing: implementation and local evidence

The compact-coordinate candidate is implemented in shared C++, with the default
areal initializer unchanged. It substantially improves the matched 20–40M
anchor's exterior accuracy at fixed point count, but does not yet pass the
differential auxiliary boundary gate. No candidate is admitted for 3D or boosted
use. All experiments are local CPU runs in the isolated optimization worktree.

## Geometry and equations

`CentralZ4cConfiguration::compact_slicing_end` (laboratory `--compact-end`) is
zero by default. A positive value b selects `NativeWormhole`'s compact profile.
The original areal configuration first supplies the inner compact radius a and
isotropic scale k. Its central radius continues to determine the conformal,
gauge and restoration joins. The optional areal ratio is the **anchor** ratio;
it is not the physical radius of the new outer join.

For z=(r-a)/(b-a), S=I_z(p,7), with exact zero/one plateaus. Shape p is solved
in [7,150] to match y(a)=1/R0, where

```
y = 1/R
F = sqrt(y*y - 2*M*y*y*y + S*S)
dy/dr = -F/r
a_s = F/y
chi = (r*y/Omega)^2
Ar = -2*r*S_r/(3*Omega*a_s)
K = -3*S-r*S_r/a_s
```

The outer boundary value y(b) comes from the unchanged S=1 CMC map. Independent
adaptive RK4 step-doubling integrations, with Richardson extrapolation, supply
each requested radius. There is no sampled field interpolation or evolution
constraint solve. The second integrated quantity is the original free shift
integral, with derivative `-alpha_physical*S_r*y/F`. The actual free initial
lapse is preserved; the stationary source lapse is separately `Omega*a_s`.
`StationaryGaugeReference` uses this same map, inferring S and S_R from K and Ar.
The spherical adapter retains the nonstationary central telegrapher RHS.

For R0/M=20, anchor ratio 2, b=.97:

| Quantity | Value |
|---|---:|
| Inner compact radius a | .6402906611935943 |
| Inner isotropic scale k | .10116863541976967 |
| Solved p | 23.0688249035775 |
| Outer join R/M | 98.47738926561057 |
| Relative central matching residual | 9.33e-15 |

All joins are at least C6. Impossible ordering and shapes without a bracket
are rejected; e.g. the same b=.97 cannot preserve the 6M anchor within this
shape family. `NativeBoostedWormhole` explicitly rejects the option because its
Tail/Fields/Radial construction is not adapted. The refined boosted parser
also rejects it. The option has not been silently exposed in a 3D initializer.

## Verification before evolution

The default and compact CTests each pass 198 nontrivial off-constraint
spherical/Cartesian jet comparisons, including changing geometry, independent
Lambda/Theta/D and nonstationary boundary fixtures. Compiled default initial
data, sources and RHS match the previous Release library exactly on 257 jets.

`test_compact_profile.py` compares the shared constructor with the handoff's
independent SciPy beta-function/DOP853 construction, also independently
integrating the free shift. The maximum scaled field disagreement is 9.30e-12;
tightening the C++ map tolerance from 2e-13 to 3e-14 changes fields by at most
7.50e-12. The lapse first derivative agrees to 1.90e-10 scaled. Independent
analytic H is below 8.36e-15. Stationary outer gauge RHS components cancel to
below 1e-10 using independently constructed jets; finite-differenced geometric
RHS residuals are below 9.1e-7. The actual core RHS is nonzero, with maximum
35.09 in the checked fixture. Geometry plots were visually inspected.

The first integrator setup failed because its local error budget was multiplied
by step width again and could fall below roundoff. Its source and failure are
preserved in `compact-setup-failure-v14`. The corrected constructor uses a
local absolute/relative error budget and passes tolerance refinement. A second
CTest assumption incorrectly required the 6M anchor to match; that input is
now explicitly checked as a rejection. An analysis attempted before the N256
run completed is preserved in `compact-1M-comparison-v14/failure.json`.

## Evolution evidence

Controls use the ten-variable algebraic chart, separated gauge speeds,
reference derivatives at h=.000125, exact scri reference jets, analytic pole
normal, uniform grid, unchanged KO=.5 and penalty=.6510416666660156. No floors
are enabled. Physical H/M/Theta/Z are independently evaluated on reference
R/M=25–100; the indicator below is their largest dimensionless log-R RMS.
Interpolation disagreement is recorded in the comparison JSON files.

| N | Compact error at 1M | Compact error at 10M | 10M D differential defect | 10M horizon mass |
|---:|---:|---:|---:|---:|
|64|1.95056e-3|2.96456e-3|150.697|.251585|
|128|2.28642e-4|5.77522e-4|5.24634|.344052|
|256|1.04028e-5|5.47930e-5|5.93326|.334554|

At N128, matched original 20–40M slicing gives 3.95853e-3 at 1M and
2.97582e-2 at 10M: the compact candidate improves these indicators by 17.3x
and 51.5x without changing the central scale. The earlier areal 20–80M family
had a smaller 1M error, but a much larger 10M error; it also changes the central
compact scale and must be distinguished from the matched-anchor comparison.

Independent field differences on exact reference-map coordinates decrease
over N64/128/256. The 10M largest adjacent RMS differences are .68584 and
.022091; fieldwise log2 ratios range 2.88–5.84. These coarse three-grid ratios
are not an asymptotic-order proof. At N256 the lapse/shift gradient half-maxima
cover 15 points; Lambda/D cover four, compared with two for the wider areal
control. Remaining physical H and M peaks lie near r=.945 and .859. Unsmoothed
plots and profiles are retained in `compact-localization-n256-v14`.

The N256 clock residual magnitude is .00604 and the pole maximum .00839.
The differential D defect is not converging monotonically yet. Algebraic D
materialization and exact nullness do not establish its differential PDE.
A finer grid and boundary/perturbation controls are required before admission.

The first N512 10M-target control at C_char=.4 fails at 3.14268M, at the
innermost puncture node r=.000977517. It does not supply a 10M boundary result.
At the last accepted state chi=2.21e-10 and its rate is -8.34e-6. The centered
continuum contribution is -1.54e-5; KO contributes +7.02e-6 and thus opposes
the crossing. Biased and centered advection derivatives coincide at this row.
The one-sided chi derivative is -.015705 despite the increasing first nodal
values, with beta=.00090935. These saved terms motivate checking consistent
puncture reference variables and their derivative chain rules; they do not
justify a floor or a claim that KO caused this failure. The failure and rate
decomposition are preserved in `compact-n512-10M-cfl0.4-v14` and
`compact-puncture-failure-v14`.
Halving the timestep to C_char=.2 reproduces the failure at 3.1426783829078M
(versus 3.1426783829082M). Both last states and the comparison are preserved.
Thus no N512 10M endpoint is available, and timestep reduction is not an
effective remedy for this direct-variable puncture discretization.

## Characteristics and RK4

Full 26-field radial, transverse and oblique point symbols are sampled through
10M for N128. No defective clusters are found at those samples; maximum
imaginary part is 5.85e-16 and eigenspace condition reaches 1.73e4. This is an
interior sampling result, not uniform endpoint hyperbolicity or nonspherical
stability. The actual initial maximum coordinate speed is 2.41421356, not one.

The default nominal characteristic target .2 is capped by the scri policy:
actual accepted C_char≈.15451. For N64, the separate explicit-timestep sweep
completes 10M at targets .1,.2,.3,.4,.5,.6; .65 fails at .80136M and .7 at
.41095M. Failed-stage and last-accepted arrays are preserved. Exterior field
RMS differences relative to .1 range from 7.71e-9 at .2 to 6.75e-7 at .6.
The frozen initial decaying-spectrum RK4 bound is .606906.

Initial and 10M N64 semidiscrete Jacobians agree under finite-difference
refinement to relative 2.33e-11 and 3.21e-11. Their largest real eigenvalues
are .64049 and .59332. The disclosed scaled operator norms are strongly
nonnormal (one-step norms in the hundreds); positive modes and nonnormality
preclude interpreting the decaying-spectrum bound as a contraction theorem.
The .6 completion is not a robust timestep recommendation. Perturbed and
longer-duration checks with an accuracy margin remain required.

At N128, .5 and .6 also complete 10M, while .65 fails at .51314M. Their
constraint indicators agree with the quarter-default-step control to 1.7e-12
absolute. Default/half/quarter-step common-exterior field differences are
6.44e-11 and 2.07e-12. The lapse's adjacent ratio corresponds to order 4.13;
several smaller differences are near rounding sensitivity, so this is chiefly
an upper bound on temporal contamination, not a precise all-field order fit.
The differential D defect remains 5.24634 under timestep refinement: a smaller
step does not resolve the boundary gate.

## Reproduction and next decision

Build from this worktree with the existing CMake Release recipe. The frozen
v14 library, flags, sampler and source archive are in
`evidence/compact-equivalence-v14`. Its SHA256 is
`cfbd91aad476cae2de7ff65eae4ef63499e1839076123271f4013b5303e7cb3c`.
Each evolution includes exact CLI configuration, source archive, source/binary
hashes, setup/stepping time, profiles and failures. For example:

```sh
build-spherical/python/bin/python tst/spherical_optimization/run.py \
  --library build-spherical-v14/libcurrent_spherical.dylib \
  --output NEW_DIRECTORY --n 128 --compact-end .97 --separated \
  --algebraic-chart --reference-jets --reference-step .000125 \
  --reference-tail scri --normal-policy analytic --duration 10 --cfl .2
```

Use `test_compact_profile.py` with `--sampler build-spherical-v14/sample_compact`
for geometry; use `analyze_chart.py`, `analyze_convergence.py`,
`analyze_timestep.py`, `analyze_localization.py`, `audit_semidiscrete.py` and
`audit_evolved_symbols.py` on saved cases. Output directories must be new.
Timing observations are exploratory: independent analyses and stability
controls overlapped some runs. Do not present these as a controlled fixed-error
runtime frontier. Setup is separate from stepping, and compiler effects remain
separate from formulation comparisons.

The next decision is whether the promising exterior improvement survives
finer boundary resolution, consistent variable/source controls and regular
perturbations. Complete those gates and controlled timing before selecting a
robust spherical operating point. Small spherical 3D comparison, nonspherical
perturbations, interfaces/MPI and the actual boosted test remain separate
future transfer work; no Aurora jobs were launched.
