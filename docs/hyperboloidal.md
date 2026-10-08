# Experimental hyperboloidal Z4c prototype

## Status

The branch now contains a **single-core spherical conformal Z4c evolver**, using
the full Cartesian tensor RHS with analytic spherical angular derivatives. It
includes CMC Minkowski reference geometry, live reference gauges, Schwarzschild
trumpet initial data, constraint/mass/horizon diagnostics, and a staggered scri
boundary. A 128/256/512-cell live-puncture study reaches t=5 (100M) with decreasing
constraint and field errors. Global constraint convergence is slow near the
puncture; this is finite-duration evidence, not a long-time stability result.
See the final section for the successful parameters and measured limitations.

**Converged long-time Cartesian puncture evolution is not yet established.**
The default AthenaK path remains Cauchy. An opt-in, single-block vacuum
hyperboloidal runtime now dispatches native RK tasks, masked algebraic projection,
physical ADM conversion and constraint diagnostics. It supports CMC Minkowski
and trumpet initial data, a lapse pulse, output masks and restart. See the native
runtime section below for exact configuration and restrictions. The earlier
standalone patch and spherical executables remain useful independent tests.
GPU/MPI testing is out of scope. The sections below retain the equation derivations and earlier failed
experiments so their limitations and subsequent fixes remain auditable.

Reference Einstein source values must not be used as sources for a perturbed
metric. The nonlinear kernel recomputes geometry and exposes scri pole numerators;
its interior assembly never floors Omega or evaluates at Omega=0.

Base: `HengruiZhu99/athenak`, `project/z4c_overhaul`,
`e9a2039e93ba9737b86b67e10d8f746342770431`.
The earlier conversation's preflight report was available, but its sandbox ZIP and
patch were not retrievable through the conversation attachments. These files were
implemented and tested afresh; the old report's test counts are not claimed here.

## Reference and variable conventions

Use signature (-,+,+,+) and K_ij = -(1/2) L_n gamma_ij. With scri radius S>0,
hyperboloid scale a>0, R=r/Omega, and T=t+sqrt(R^2+a^2),

```
Omega       = (S^2-r^2)/(2 a S)
alpha_bar   = (S^2+r^2)/(2 a S)
beta^i      = -x^i/a
gamma_barij = delta_ij
K_bar       = -3/(a alpha_bar)
K_phys      = -3/a
```

`CMCReference<T>` evaluates the regular Cartesian fields and derivatives without
division by r or Omega. Call `Validate()` on the host before passing a reference
into a Kokkos kernel. Coordinates and box bounds must be finite; box bounds must
be ordered. The polynomial reference has an analytic extension outside scri, but
that fact does not authorize evolving the physical system there.

The Penrose factor Omega and Z4c chi have different roles. For the branch's usual
chi=psi^-4 convention, the kinematic maps (before specifying Z4 constraint scaling)
are

```
alpha_phys = alpha_bar/Omega
chi_phys   = Omega^2 chi_bar
A_physij   = Omega A_barij       (Z4c trace-free variables)
K_phys     = Omega K_bar - 3 beta^i d_i Omega/alpha_bar
```

Physical lapse and metric conversion is singular at scri. The helper deliberately
does not populate physical ADM arrays there. For the reference, chi_bar=1,
g_tildeij=delta_ij, A_barij=0 and Gamma^i=0.

The unchanged vacuum RHS cannot evolve these regular fields: its Hamiltonian would
be `2 K_bar^2/3`, which is nonzero. Conformal Einstein terms must supply it.
`CMCPoint` includes the analytically simplified reference Einstein projections
E=G_nn, J_i=-G_ni, S_ij=G_ij (all with 8*pi absorbed):

```
E     = 3/(a^2 alpha_bar^2)
J_i   = -2 d_i alpha_bar/(a alpha_bar^2)
S_ij  = [-E + 2/(a S alpha_bar)
         + 2 r^2/(a^3 S alpha_bar^3)] delta_ij
```

They are finite at the center and scri. The independent symbolic test constructs
the four-metric, connection and Einstein tensor, checks all ten conformal Einstein
components, and checks these projections. This is a reference-solution check only.

## Factored gauge

`ReferenceGauge` algebraically extends equations (9) and (10) of
[Vano-Vinuales and Valente, arXiv:2408.08952v2](https://arxiv.org/abs/2408.08952)
to Cartesian advection/contractions. Their nonlinear evolution evidence is spherical;
this extension is not a validation of the full 3D characteristic structure.

The inputs are explicitly factored deviations:

```
alpha = alpha_ref + Omega L
beta  = beta_ref  + Omega B
Delta(K_phys - 2 Theta_phys) = Omega Q
```

`GaugeDeviation.trace` is Q, not AthenaK's existing Khat. Derivatives in the input
are derivatives of L and B. The kernel returns the RHS of the **unfactored** lapse
and shift. Products that would be subtracted and divided by Omega are expanded
analytically; it never divides by Omega. At the exact reference the returned RHS
vanishes exactly, including in single precision and at scri.

Finiteness alone does not close an evolution system for L, B, Q. The assumed
falloff fixes alpha and beta at scri, so their RHS must also vanish there. The
kernel gives the following compatibility relations for this CMC reference:

```
alpha_rhs|scri = -alpha_ref^2 Q
                 -alpha_ref grad(Omega).B - 2 xi alpha_ref L = 0
beta_rhs|scri  = alpha_ref B/a + (3/4) alpha_ref^2 chi Lambda = 0
```

The tests deliberately exercise incompatible data and require a nonzero residual.
They also check compatible data and approach the limit from the interior. Dividing
these returned RHS values by Omega to evolve deviations would require another
regularity derivation. No such division or boundary projection is provided.

The existing telegraph lapse is not included in this gauge kernel. Nor is there a
proof of strong hyperbolicity for the coupled metric, lapse, shift and constraint
system. A superluminal gauge speed can be incoming at scri even when physical
characteristics are tangent/outgoing. Source subtraction does not change that
principal-part problem.

## Spherical boundary treatment

No Cartesian spherical mask is introduced into the production mesh. Two regression
counterexamples make clear why naive one-sided derivatives are insufficient:

* At (S,0,0), all nonzero y- or z-offset stencil points are outside the sphere.
  There is no interior collinear stencil in either tangential direction.
* At x=y=z=S/sqrt(3), the true radial incoming light speed is zero, whereas the
  speed through a positive-x coordinate face is S/(a sqrt(3))-S/a < 0.
  A stair-stepped mask has incoming numerical faces.

`ClassifyBox` supplies geometric classification only. It makes no derivative or
outflow claim for cut cells.

The implemented **host-only model prototype** instead uses a boundary-fitted radial
shell with 0<r_inner<S. It evolves the outgoing characteristic

```
q_t + c_+(r) q_r = 0,        c_+(r) = (S+r)^2/(2 a S).
```

This is the outgoing part of the spherically reduced flat-space scalar wave;
it is not a gravitational or constraint perturbation. There are no angular
derivatives, center closure, or Cauchy/hyperboloidal layer matching in this model.
The ingoing physical speed is `c_-(r)=-(S-r)^2/(2 a S)`, evaluated in factored form
by the reference helper, but no ingoing evolution system is implemented here.

`RadialSBP` uses the second-order diagonal-norm summation-by-parts derivative:
centered differences inside, first-order endpoint closures, and trapezoid weights
H. A simultaneous-approximation-term penalty imposes the one incoming datum at
the inner shell endpoint. Scri is an evolved outer endpoint with no supplied data
or exterior ghost values. The closures are valid because the coordinate lines
follow the sphere's normal, unlike a Cartesian mask.

For C=diag(c_+), the semidiscrete energy satisfies exactly

```
E = (1/2) q^T H C^-1 q
dE/dt = -(1/2) q_N^2 -(1/2) (q_0-g)^2 +(1/2) g^2.
```

Tests verify this identity for arbitrary seeded data. RK4 at dt<=0.2 dr/max(c_+)
is additionally tested numerically; the semidiscrete identity alone is not a
fully discrete stability proof. The outgoing pulse uses the exact retarded phase
`u=t+a(S-r)/(S+r)`, including u=t at scri. Tests compare the time-dependent signal
at scri rather than just checking that evolved values remain finite.

## Reproduce validation

From the repository root (C++17, CMake, and initialized Kokkos submodule):

```sh
git submodule update --init
cmake -S . -B build -DAthena_ENABLE_HYPERBOLOIDAL_TESTS=ON
cmake --build build -j 6
ctest --test-dir build --output-on-failure
python3 tst/hyperboloidal/check_equations.py   # requires SymPy
```

The CTest executable includes runtime checks that remain enabled in Release builds.
The symbolic script exits with an error on a failed equality. To run the selected
existing regressions (requires numpy, pytest and h5py):

```sh
ATHENA_OVERHAUL_EXE="$PWD/build/src/athena" python3 -m pytest -q \
  tst/test_suite/z4c/test_z4c_overhaul_cpu.py \
  tst/test_suite/z4c/test_z4c_conversion_cpu.py \
  tst/test_suite/z4c/test_z4c_restart_cpu.py
```

Run the existing AMR/outflow smoke input in a separate run directory, then check
column 6 (L-infinity) of `lwave_z4c_bc-errs.dat` against the existing regression's
threshold 1e-12:

```sh
mkdir -p run-amr
cd run-amr
../build/src/athena -i ../tst/inputs/lwave_z4c_bc.athinput
```

For the building-block sanitizer test, configure another build with
`-DCMAKE_BUILD_TYPE=Debug`, `-DAthena_ENABLE_HYPERBOLOIDAL_TESTS=ON`,
`-DCMAKE_CXX_FLAGS="-Wall -Wextra -Werror -pedantic -fsanitize=address,undefined -fno-omit-frame-pointer"`,
and `-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"`; build the
`hyperboloidal_tests` target and run CTest.

### Results, 2026-10-06

Local AppleClang 21.0.0, C++17, Kokkos 4.7.2 Serial, double-precision AthenaK:

| Check | Result |
| --- | --- |
| Full AthenaK Release build | Passed |
| Reference/gauge Kokkos kernels, float and double | Passed on Serial |
| Factored vs unfactored gauge away from scri; scri compatibility | Passed |
| Reference constraints, stationary curvature, geometric counterexamples | Passed |
| Independent symbolic ten-component Einstein/source audit | Passed |
| Arbitrary-data SBP energy identity | Passed |
| Pulse through scri, 160/320/640 radial intervals | Passed |
| Random-data energy over 20 shell-crossing times | Passed |
| Debug building blocks, warnings as errors, ASan and UBSan | Passed |
| Changed C++/Python files, repository cpplint/flake8 rules | Passed |
| Existing overhaul/conversion/restart CPU regressions | 46 passed |
| Existing 32^3 AMR/outflow Z4c linear-wave input | Passed, 15 cycles |

Maximum pulse-signal errors at scri were 6.89368e-3, 1.72004e-3, 4.29766e-4,
consistent with second-order convergence. Final shell energies were 8.05e-10,
4.55e-11, 2.78e-12. Random-data energy fell from 0.115603 to 2.39e-19 without
detected growth. The existing AMR regression's L-infinity column was 2.362895e-14,
below its 1e-12 threshold. Those existing tests exercise ordinary Cauchy Z4c;
they do not exercise hyperboloidal gravity.

An initial 80-interval pulse run missed the 0.025 accuracy target (error 0.0276684).
The accuracy target was retained and the convergence sequence refined to
160/320/640 intervals. No assertion was disabled to obtain the reported passes.

## Work still required for the requested solver

1. Integrate the experimental off-constraint conformal Z4c interior tensor kernel
   into a time evolution. Reference Einstein sources cannot replace its nonlinear
   geometric terms. Prove the limiting combinations at scri and verify them on
   nontrivial data; the existing interior assembler explicitly refuses Omega=0.
2. Close the gauge and metric limiting equations together. Check the complete
   coupled principal symbol and compatible initial data, not only isolated lapse
   or radial physical speeds. Arbitrary perturbations of factored fields need not
   satisfy the boundary regularity relations.
3. Implement a 3D boundary-fitted outer region, a genuine multidimensional embedded
   boundary method, or a mathematically justified extension through scri. Integrate
   it with every RK stage, physical boundaries, constraints, ghost exchange,
   restriction/prolongation, timestep calculation, ADM conversion and output.
4. Validate nonlinear gauge/constraint perturbations and convergence before black
   holes, punctures, matter or AMR at scri. GPU, MPI, hyperboloidal AMR, BBH and GRMHD
   have not been tested. There is no claim of a stable nonlinear Z4c prototype.

The equation/gauge reference is
[Height-function-based 4D reference metrics for hyperboloidal evolution](https://doi.org/10.1007/s10714-024-03323-8).
The implementation is an original algebraic specialization and test harness;
no published evolution code has been imported.

## Milestone 2: off-shell constraint diagnostics

`conformal_constraints.hpp` now evaluates Cartesian spatial geometry and vacuum
physical ADM constraints from the Penrose-rescaled spatial metric and extrinsic
curvature, including their spatial derivatives. These kernels accept arbitrary
data; they do not subtract the Minkowski reference and do not assume the constraints
are satisfied. They are not yet connected to production field output.

Let `w = n_bar(Omega)`, `b = gamma_bar`, and `k = K_bar`. Write `D`, `R`, tensor
contractions, and `div(k)` using b. Then the implemented identities are

```
gamma_phys = Omega^-2 b
K_physij   = Omega^-1 k_ij + Omega^-2 b_ij w
K_phys     = Omega tr(k) + 3 w

H_phys = Omega^2 [R + tr(k)^2 - k_ij k^ij]
         +4 Omega [D^2 Omega + w tr(k)]
         -6 [|D Omega|^2 - w^2]

M_phys_i = Omega [D_j k^j_i - D_i tr(k)]
           -2 k^j_i D_j Omega -2 D_i w.
```

For fixed Omega, `w=-beta.grad(Omega)/alpha`; the helper also evaluates its spatial
gradient from lapse/shift derivatives. Neither physical constraint formula divides
by Omega. The reported null residual is `|D Omega|^2-w^2`. Its vanishing is required
at scri, not throughout the domain.

Both momentum norms are retained: `b^ij M_i M_j` and the physical norm
`Omega^2 b^ij M_i M_j`. Reporting only the latter would hide some boundary violations.
Likewise, the spatial Z diagnostic retains its unweighted covector and conformal
norm alongside its physical norm. The Z4 helper takes the twice-conformal spatial
metric and computes

```
Z_i = (1/2) g_tilde_ij [Lambda^j - contracted_Gamma(g_tilde)^j]
determinant_residual = det(g_tilde)-1
tracefree_residual   = g_tilde^ij A_ij.
```

This expression assumes the Cartesian CMC reference's zero spatial connection.
Theta must be supplied in its physical normalization explicitly. The helper does
not silently reinterpret the existing AthenaK evolution variable. Invalid spatial
metrics and nonfinite results produce a false validity flag; the physical
constraint helper also returns a NaN Hamiltonian for an invalid geometry.

The `hyperboloidal_constraints` CTest adds:

* CMC Hamiltonian/momentum identities along a non-axis-aligned radius including
  the origin and scri, evaluated in a Kokkos Serial kernel.
* A non-flat, non-diagonal manufactured metric with nonzero curvature shear and
  nonzero constraints. Its Ricci scalar is checked against the analytic conformal
  transformation of a constant metric. A separate finite-difference path constructs
  the physical ADM fields and checks convergence to the conformal constraints.
* Incompatible data at scri that must remain visible in H and unweighted M.
* Nonzero Theta, spatial Z, determinant and trace-free diagnostics.
* Exact vacuum Hamiltonian/momentum checks on time-symmetric isotropic Schwarzschild
  data away from the puncture. This tests diagnostics, not hyperboloidal puncture
  evolution.

For the two manufactured points, halving spacing from 0.02 through 0.0025 reduces
Hamiltonian and momentum errors by approximately four on each step. Final H errors
are 6.51e-5 and 1.90e-5; final maximum component M errors are 5.06e-6 and 9.40e-6.
Both hyperboloidal CTests pass in Release and in the strict-warning ASan/UBSan Debug
build on one Serial execution thread. This milestone changes no production RHS.

The next evolution implementation is based on the general tensor equations in
Appendix B of [arXiv:1412.3827](https://arxiv.org/abs/1412.3827), with the physical
trace/Theta transformation in section 7. The source explicitly identifies an
instability in evolving the untransformed conformal trace. The target remains a
single-puncture hyperboloidal evolution with constraint, convergence and stability
tests; these diagnostics do not fulfill that target on their own.

## Milestone 3: nonlinear conformal interior RHS (2026-10-07)

`conformal_rhs.hpp` implements the vacuum tensor system with C_Z4c=0 and the
physical curvature/constraint variables

```
P = K_phys - 2 Theta_phys = Omega Khat_bar + 3 w
T = Theta_phys = Omega Theta_bar,
w = n_bar(Omega).
```

`Z4cJet.trace` stores the full P, not P minus its reference value. The other inputs
are chi, the unit-determinant metric, its trace-free curvature A, Lambda, lapse,
shift and their Cartesian spatial derivatives. `PenroseMetric` converts spatial
metric jets without changing the variable interpretation. `Geometry` additionally
provides contracted connection derivatives for computing spatial Z derivatives.

The kernel implements chi, metric, A, P, T and Lambda evolution. Lapse/shift
evolution is supplied separately by the gauge kernel; neither time integration nor
a numerical spherical boundary is included in this milestone. The equations assume
the algebraic determinant/trace-free constraints, which must be enforced by the
eventual time integrator.

The returned form is `dt(u) = regular + pole/Omega`. Both returned parts are finite
functions of regular fields, including at Omega=0. The assembler requires positive
Omega and rejects nonfinite results; it never floors Omega. The pole numerators
must satisfy compatibility conditions before a boundary limit may replace this
interior formula. In particular, a finite numerator is not a finite RHS.

Using the Penrose spatial metric b and its derivative D, B=P+2T, and
`A2 = A_ij A^ij` with the twice-conformal metric, the transformed trace equation is

```
P_t = beta.grad(P) + Omega [alpha A2 - D^2 alpha]
      +3 D(alpha).D(Omega) + alpha D^2 Omega
      +(alpha/Omega) [B^2/3 - 3 |D Omega|^2 + kappa1 (1-kappa2) T].
```

The chain rule cancels every gauge time derivative and every double pole. The
Theta equation from the tensor system retains `-3 alpha w T/Omega`. This differs
from the later stabilized spherical equations in the same paper, which omit that
off-constraint term. The kernel explicitly chooses the tensor system and tests the
coefficient; a subsequent stability experiment may motivate the published
alternative, but the two versions are not silently conflated here.

`EvolvedConstraints` evaluates H, M, Theta, Z and algebraic residuals from these
evolved variables directly. In particular, H and M no longer depend on w:

```
H_phys = Omega^2 [R(b)-A2] + (2/3) B^2 +4 Omega D^2 Omega-6 |D Omega|^2
M_phys_i = Omega [Dtilde_j A^j_i - (3/2) A^j_i d_j(log chi)]
           -(2/3) d_i B -2 A^j_i d_j Omega.
```

Thus diagnostics at scri do not reconstruct `K_bar=(B-3w)/Omega`. Tests compare
them with the previous ADM-jet diagnostics off the constraint surface and verify
that incompatible data at scri remain visible.

The new `hyperboloidal_rhs` CTest verifies:

* The complete interior geometric RHS vanishes on CMC Minkowski, using a Serial
  Kokkos kernel along a non-axis-aligned radius.
* Reconstructing ordinary ADM metric and curvature evolution from the Omega=1,
  Z=Theta=0 limit agrees with independent ADM equations on data with nonzero H.
  Theta must respond to the Hamiltonian violation in this test.
* All RHS components approach zero at second order on a different exact solution:
  the stationary Schwarzschild CMC exterior with M=0.05, K_phys=-3 and C=0,
  compactified using the Minkowski Omega. Finite differences sample the exact
  non-flat solution; there is no Schwarzschild reference subtraction.
* Nonzero Theta/spatial-Z damping has the physical normalization and the retained
  off-constraint w term is exercised. Zero/negative Omega assembly, overflowing
  results and negative chi are rejected.

At compactified radii 0.3, 0.65 and 0.85, halving stencil spacing from 0.004 through
0.0005 reduces the maximum stationary RHS residual by approximately four per step.
The finest residuals are 4.97e-6, 1.56e-7 and 3.50e-8. These are pointwise spatial
consistency tests, not time-evolution stability or global convergence results.

`python3 tst/hyperboloidal/check_rhs_transform.py` independently checks the trace,
Theta, A and Lambda variable transformations with SymPy. It keeps lapse and
compactifier time derivatives independent and verifies their cancellation, rather
than assuming stationarity. All three CTests pass in Release and strict-warning
ASan/UBSan Debug builds on one Serial thread. Changed files pass repository lint.

Next: connect this RHS and gauge to a boundary-fitted time-evolution driver, test
the scri limiting/staggered treatments on constraint and gauge perturbations, and
then construct and evolve Schwarzschild trumpet/puncture initial data. The C=0
Schwarzschild exterior test above is not such a puncture and does not complete the
active single-puncture goal.

## Standalone spherical evolution experiment

`hyperboloidal_spherical` now time-integrates the nonlinear tensor kernel on one
Kokkos host thread, with live reference lapse and shift. This is a standalone
spherical experiment, not the AthenaK 3D task graph or a puncture run. Units are
S=a=1. It stores deviations from CMC Minkowski for chi, the radial conformal
metric, physical trace, lapse and shift, plus A_rr, physical Theta and Lambda^r.
The angular metric and curvature enforce unit determinant and zero A trace.
Analytic Cartesian angular derivatives of spherical scalars, vectors and tensors
are included; independent Cartesian polynomial tests exercise these derivatives.

The grid is cell-centered on 0<r<1. The origin uses parity ghosts and scri uses
polynomial continuation of the deviations. All evolution points have Omega>0;
there is no Omega floor and no RHS evaluated outside scri. This avoids intersecting
Cartesian stencils with a sphere, but does not supply the exact Omega=0 limiting
equations. Fourth-order centered differences, RK4 and sixth-difference dissipation
are used. The analytic Minkowski RHS is subtracted to remove its floating-point
residual; perturbed fields use the full nonlinear equations. Default outer
polynomial degree is four; degrees three and five are comparison treatments.

Outputs include radial L2 norms of H, M, Z, Theta, maximum H/M, minimum chi/lapse,
maximum state deviation, and the null residual at the last interior point. Norms
use dr, without r^2 or Omega weights that could hide center or scri errors. Final
snapshots include every evolved field, H, M_r, Z_r and both physical radial light
speeds. These speeds do not constitute an audit of every gauge/constraint mode.
The last interior null residual is not an exact scri boundary diagnostic.

The default perturbation is a compact smooth lapse pulse of amplitude 0.001,
center 0.4 and half-width 0.12; the initial geometric constraints vanish. With
CFL 0.05, dissipation 0.1, kappa1=1.5 and live shift, the t=1 results are:

| Radial cells | H L2 | M L2 |
|---:|---:|---:|
| 64 | 3.5063941e-3 | 2.3827287e-3 |
| 128 | 8.8908305e-4 | 1.0548326e-3 |
| 256 | 3.3541561e-4 | 3.7829432e-4 |
| 512 | 1.09362e-4 | 1.65388e-4 |

All eight evolved fields also show decreasing self-differences at t=1 after
fourth-order interpolation onto matching radii. The finest-pair orders range
roughly from 1.5 to 2.3, not four. This test therefore establishes error reduction
for this pulse, not the nominal fourth-order global convergence of the scheme.
The 512-cell run has an intermediate H L2 peak near t=0.3; final norms alone
must not be interpreted as uniform-in-time fourth-order convergence. At t=1 the
largest H/M errors are in the interior, while Z has its maximum near scri.

At 128 cells the run reaches t=10 with H L2=7.42e-12, M L2=6.94e-12 and maximum
state deviation 2.13e-7. Cubic and quintic outer continuation both reach t=3;
H/M L2 are respectively (8.45e-5, 6.80e-5) and (6.98e-5, 3.79e-5). These are
finite-duration stability observations, not an energy estimate or proof of
stability under arbitrary boundary or constraint perturbations.

Reproduce the small reference/pulse checks with CTest, or the full experiment:

```sh
python3 tst/hyperboloidal/check_evolution.py build/hyperboloidal_spherical \
  --extended --output evolution-results
```

The script checks exact reference preservation, finite positive geometry, pulse
decay, decreasing constraint norms and all eight field differences, the longer
run, two alternative boundary closures, and rejection of nonfinite CLI inputs.
The thresholds require error reduction but intentionally do not claim fourth
order. Raw diagnostic histories and field snapshots are retained under --output.
The unresolved order reduction, exact scri compatibility and trumpet initial data
remain work toward the single-puncture objective. No black-hole evolution is
claimed by this experiment.

Validation for this milestone: all four Release CTests pass, the extended driver
script passes, and the three kernel CTests plus a 16-cell t=0.05 pulse pass under
ASan/UBSan with strict compiler warnings. The long and resolution-study runs were
Release-only. C++ and Python lint pass. The production AthenaK evolution path is
unchanged; its earlier Cauchy regression results are not hyperboloidal evidence.

## CMC trumpet initial data and failed gauge experiments

`cmc_trumpet.hpp` constructs a Schwarzschild CMC trumpet with mass M>0, S=a=1,
K=-3. With physical areal radius R, define J=-R+C/R^2 and
D=1-2M/R+J^2. The critical C is obtained from the double zero of D at R0 between
1.5M and 2M. The isotropic compact coordinate satisfies

```
log(1/r) = integral_R^infinity dR / (R sqrt(D)).
```

The code integrates u=1/R against t=-log(r), starting at u=0 at scri. It factors
the double zero analytically before evaluating the square root. This avoids a
finite areal-radius cutoff and cancellation at the cylindrical end. The fields
are chi=(r u/Omega)^2, A_rr=-2 C u^3/Omega, alpha=Omega sqrt(D),
beta^r=r(-1+C u^3), unit conformal metric, P=-3, Theta=Lambda=0. Analytic first
and second spatial derivatives accompany the initializer. Omega and the live
gauge reference remain the Minkowski compactifier and reference used above.

Independent checks verify the double root, the limiting areal radius at r=1e-6,
quadrature refinement, analytic constraints and stationary geometric RHS below
1e-8 at r=0.02, 0.1, 0.5 and 0.9. A separate finite-difference reconstruction
converges at second order on these points. These checks do not subtract a
Schwarzschild evolution residual. Misner-Sharp mass reconstructed from geometry
and curvature agrees with M=0.05 within 1e-10, and the outgoing null expansion
changes sign at areal radius 2M. These are initial-data/equation tests.

The driver accepts `--mass 0.05 --amplitude 0` to initialize this trumpet. Its
output now includes areal radius, Misner-Sharp mass and outgoing expansion in
field snapshots, mass near coordinate r=0.5 and an interpolated outermost apparent
horizon areal radius in time histories (zero if no crossing is resolved). The
horizon value is a linear interpolation of expansion, not a high-order finder.

For the finite-difference initial data, including every cell in the norms:

| Cells | H L2 | M L2 | Mass near r=0.5 | Horizon areal radius |
|---:|---:|---:|---:|---:|
| 128 | 11.2402 | 8.01055 | 0.0499999757 | 0.1001911 |
| 256 | 2.98715 | 2.00407 | 0.0499999985 | 0.1001775 |
| 512 | 0.739734 | 0.487410 | 0.0499999999 | 0.1000378 |

The large constraint errors are concentrated at the puncture end and decrease
roughly quadratically in these unweighted norms. They must not be confused with
the tiny residuals obtained using analytic derivatives. The initial-data
resolution checks and mass/horizon tolerances are part of the evolution CTest.

**Stable puncture evolution is not established.** The following single-core
experiments expose failures, rather than supplying a passing puncture gate:

* The original live reference gauge at 128 cells fails after t=0.12, before 0.2.
* Setting the extra slicing and shift-driver coefficients to zero delays failure
  to shortly after t=0.30. It does not cure it.
* `--puncture-gauge` changes the lapse restoring term from
  `-xi (alpha^2-alpha_ref^2)/Omega` to
  `-xi alpha (alpha-alpha_ref)/Omega`. It preserves the Minkowski fixed point
  and lets this restoring term vanish for a collapsed lapse. With zero extra
  slicing/shift-driver coefficients, it fails after t=0.36 at 128 cells and
  after t=0.20 at 512 cells. Resolution alone therefore does not fix the gauge.
* With the lapse-weighted alternative and fixed shift, 256 cells fail after t=0.4.
  Fixed lapse with a live shift and zero extra shift coefficient also fails
  after t=0.3. Holding only one gauge component fixed is insufficient.
* Holding both lapse and shift fixed reaches t=1 at 256 cells, but has H L2=3.34,
  M L2=1.97 and growing field drift. This is not a demonstrated stable equilibrium.

`--fixed-lapse` and `--fixed-shift` support these isolation experiments. They do
not manufacture an evolved stationary source. No puncture lapse/chi floor or
interior excision is used. The next work is to control the puncture-end spatial
errors and audit the live gauge/constraint modes before accepting a long-time
black-hole evolution. The new data and diagnostics are verified; these gauge
options remain experimental and the failed runs are not regression successes.

Validation: all four Release CTests pass after these additions. The three kernel
CTest cases and a 32-cell fixed-gauge t=0.01 trumpet smoke run pass under strict
warnings and ASan/UBSan; the updated analytic trumpet test was rebuilt and rerun
under sanitizers after its final assertions were added. C++/Python lint pass.
Sanitizer success checks memory/undefined behavior, not physical stability; the
coarse sanitizer smoke run itself has large constraint errors.

## Analytic-profile reconstruction and live puncture gauge

The optional `--analytic-trumpet` treatment differentiates the difference between
the evolved fields and the initial analytic trumpet profile, then adds the exact
profile derivatives. Dissipation acts on that difference as well. The initial
profile and its derivatives are computed once. This changes the spatial
approximation, not the continuum nonlinear equations: no Schwarzschild RHS is
subtracted or frozen. The reference compactifier and gauge source geometry remain
CMC Minkowski. This option is specialized to the analytic spherical trumpet; it
is not a generic black-hole or binary reconstruction method.

With both gauge variables fixed, the 128-cell trumpet remains near roundoff
through t=1 (H L2 about 5e-12, M L2 about 1.2e-11), unlike the original derivative
scheme. The automated test exercises the same equilibrium at 64 cells. It also
checks that the independently computed **plain finite-difference** constraint
norms remain large and unchanged: `H_raw_L2` and `M_raw_L2` are now always output
beside the reconstruction-based norms. Equilibrium preservation alone is not a
stability or convergence test for a perturbed black hole.

This reconstruction does not by itself fix the live-gauge failures. Parameter
experiments with constant slicing coefficients, a fivefold smaller timestep and
stronger dissipation still failed on the finer grid. Two additional optional
changes are used in the subsequent live-gauge study:

* `--one-plus-log` makes the extra slicing coefficient proportional to lapse:
  `alpha^2 + slicing * alpha * (1-r^2)^2`, replacing
  `alpha^2 + slicing * (1-r^2)^2`. Thus the additional term vanishes with the
  puncture lapse; the outer gauge still tends to harmonic slicing.
* `--lapse-scaled-damping` passes `kappa1/alpha` to the tensor kernel so the
  coordinate-time damping coefficient remains active where lapse collapses.
  Every evolved grid point must still have positive lapse. There is no lapse
  floor, and the existing continuum kernel is unchanged.

The tested choice additionally uses `--puncture-gauge` (the lapse-weighted
restoring source), slicing=2 and shift_driver=0.1. The nonzero interior shift
coefficient avoids relying only on alpha^2 chi near the puncture. Without
lapse-scaled damping, a 256-cell run reaches t=5 but its mass near r=0.5 has
increased to 0.0592186 from 0.05: remaining finite did not constitute success.
With lapse-scaled damping and kappa1=5 the corresponding mass is 0.0499965268.

Scri diagnostics now extrapolate the fields and their derivatives to r=1 and
report `scri_null_residual`, `scri_pole_max` (all geometric pole numerators), and
`scri_lapse_pole`. No pole is divided by Omega=0. These diagnostics test
compatibility independently of the last interior point. They do not impose
boundary values or implement a limiting evolution equation at scri. The actual
integrator retains its staggered grid and polynomial outer continuation.

The live study uses M=0.05, zero added pulse, CFL=0.05, dissipation=0.1, the options
above and default quartic outer continuation. Every run evolves both lapse and
shift. At t=5 (100M in the chosen coordinate time):

| Cells | H L2 | M L2 | Plain-FD H L2 | Plain-FD M L2 | Mass near r=0.5 | Horizon R |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2.78169 | 2.48871 | 13.4181 | 9.97200 | 0.049365697 | 0.098449603 |
| 256 | 1.98582 | 1.49156 | 4.83999 | 3.43279 | 0.049996527 | 0.100228449 |
| 512 | 1.34743 | 0.976283 | 2.05981 | 1.45607 | 0.049999529 | 0.100036220 |

The reconstructed H/M norms converge slowly, at approximately 0.49--0.74 order
across these resolutions. Their largest errors remain near the puncture; those
cells have **not** been removed from the norms. Plain-FD H/M norms improve at
1.23--1.54 order, Z at about 1.64 and Theta at about two. The maxima over the
sampled time histories also decrease for H, M, Z and Theta. All eight field
self-differences decrease using fourth-order interpolation to common radii,
excluding only the two endpoint samples that need interpolation ghosts. Their
apparent orders range from 0.58 to 5.42; the high values are not evidence of
superconvergence or a clean asymptotic regime. Fourth-order global convergence
has not been demonstrated.

The final scri geometric pole maximum drops from 5.14e-7 at 128 cells to 1.78e-9
at 512 cells; extrapolated null residuals are -2.44e-9 and -1.22e-12. The 256-cell
run predates the added endpoint diagnostic columns but uses identical evolution
and reconstruction equations. Its data is used for the evolution study, not for
an unmeasured intermediate scri residual.

Reproduce the live study (sequential runs, one host thread each) with:

```sh
python3 tst/hyperboloidal/check_puncture.py build/hyperboloidal_spherical \
  --output puncture-results
```

The script checks finite positive geometry, mass and horizon error bounds,
decreasing final and sampled-maximum constraint norms, all eight self-differences,
and decreasing scri geometric pole residuals. `--prefixes` audits three existing
output prefixes without rerunning them; it does not establish their provenance.
The reported study was audited using the completed run outputs described above.
These results establish a finite-duration spherical live-gauge puncture prototype,
not long-time stability, a Cartesian mesh implementation, or a general scri
boundary energy estimate. Those limitations remain part of the active task.

For the same live-gauge parameters at 256 cells, cubic and quintic outer
continuation also reach t=5. Their final H/M norms differ from the quartic result
by less than 1.5e-6 in absolute value; the mass differences are below 1.5e-11.
Their extrapolated scri pole maxima are 3.13e-7 and 5.68e-8, respectively. This
comparison supports the numerical closure for this finite-duration spherical
experiment; it is not a proof of a general Cartesian cut-cell boundary treatment.

A control run with the same live gauge and damping **without** analytic-profile
reconstruction also reaches t=5 at 256 cells. Its H/M L2 norms are 5.66692/4.25789,
with mass 0.0500432453 and horizon R=0.100182748. Thus the gauge/damping improvement
is not limited to a numerically preserved equilibrium, although the plain scheme
has larger constraint errors and has not yet received its own resolution study.
New runs also write a `-config.txt` sidecar containing all numerical and gauge
options; the earlier study outputs predate that provenance sidecar.

Validation for this milestone: all four Release CTests and all four ASan/UBSan
Debug CTests pass with strict compiler warnings. The final diagnostic/configuration
output changes were rebuilt in both configurations, followed by another Release
CTest pass and a 32-cell live-gauge sanitizer smoke run through t=0.02 with quintic
scri extrapolation. The 128/256/512-cell convergence audit passes, as do C++ and
Python lint. The t=5 resolution/boundary studies are Release-only; sanitizer
success is not used as evidence of physical accuracy.

## AthenaK Cartesian mesh interface

`athenak_bridge.hpp` now loads the conformal tensor jets directly from AthenaK's
`Z4c_vars` field views using its existing `Dx`, `Dxx` and `Dxy` operators. It
packs the geometric and gauge RHS into the real AthenaK variable layout,
including explicitly zeroing the unused auxiliary gauge slots. The input must
already use the documented Penrose/physical-trace variable convention. This
adapter does not reinterpret ordinary Cauchy initial data as conformal data.

The protected loader requires the complete Cartesian stencil box to lie strictly
inside scri. A test constructs an interior point where every axial sample is
inside but a mixed-derivative corner is outside; rejection occurs before any
field access. Rejection requires a boundary treatment from the caller, not a
permission to freeze or drop that cell. This is an interior adapter, not a
cut-sphere closure. It also does not check allocation extents or a wider
artificial-dissipation stencil on the caller's behalf.

`ToPhysicalADM` returns independent point values of physical metric, curvature,
lapse and shift for Omega>0, and rejects scri/exterior evaluation without a floor.
The physical curvature uses P+2Theta and Omega*A. Non-diagonal, tracefree test data
with nonzero Theta verify those factors and recovery of the physical trace.

The physical lapse and evolved Penrose lapse differ by Omega. The default ADM
storage aliases Z4c lapse; the opt-in independent storage described below removes
that obstacle. The adapter intentionally returns point values. A future conformal
runtime must dispatch this conversion instead of the current Cauchy conversion;
no hyperboloidal production runtime switch has been exposed.

The live puncture gauge is now a shared, three-dimensional
`UnfactoredReferenceGauge` kernel. It returns separate regular and pole parts,
with interior-only assembly. Tests compare it with the independently factored
reference gauge on non-axis-aligned points for S=2, a=3, including near scri, and
check both puncture modifications. The spherical driver uses this shared kernel.
Repeating its 128-cell t=5 live-puncture run changes the final fields and diagnostics
by at most 4.15e-11 relative to the previous implementation.

The new Cartesian mesh CTest verifies all three AthenaK FD orders on data with
nonzero mixed derivatives and distinct field amplitudes. Halving grid spacing
reduces the summed jet errors by about 4, 16 and 64 for the second-, fourth- and
sixth-order operators. It also evaluates the complete geometric/gauge RHS and
constraints on a 6x6x6 CMC Minkowski interior with actual AthenaK views, verifies
reference stationarity, checks all RHS slots with distinct nonzero values, and
confirms that output halo sentinels are untouched. These are Cartesian volume and
interface tests, not a Cartesian puncture time evolution or a boundary stability
result. An unused overload-tag parameter in the shared task-list header was made
anonymous so this real-header test can compile under strict warnings; no task-list
behavior changed.

The next integration steps remain a validated sphere-crossing stencil/closure
and active-cell policy, and dispatch of
the conformal kernel with compatible initialization, RK, constraints and outputs.
Passing the interior adapter tests does not complete those steps.

## Longer spherical evolution

The previous successful parameters at 256 cells reach t=20 (400M), with positive
metric/lapse and no failed RHS evaluation. The largest sampled relative mass error
is 1.68645e-3. At the endpoint H/M L2 are 2.98692/2.19098, mass is 0.0500843225,
horizon R is 0.100396241, and the scri geometric pole maximum is 2.78e-6.
The maximum sampled H L2 over this run is 3.55040. Late oscillations and increased
mass error relative to t=5 motivate the finer long run; finite completion at one
resolution is not a long-time convergence result.

Validation of the mesh-interface milestone: five Release CTests pass. Four
non-evolution ASan/UBSan CTests pass, and the final adapter assertions were rebuilt
and rerun under sanitizers after adding nonzero component/value checks and
conversion-underflow rejection. A short live-puncture sanitizer run also passes
through t=0.02. The full longer spherical runs are Release-only. Changed C++ files
and Python regression scripts pass lint.

The finer 512-cell run has now also completed t=20 (400M), using the shared gauge
kernel with the same parameters. Final H/M L2 are 1.88086/1.29648, raw finite-
difference H/M L2 are 2.58348/1.76894, mass is 0.0499967438, and horizon R is
0.100002456. The largest sampled relative mass error is 6.51236e-5. The final
extrapolated scri pole maximum is 5.86566e-8 and null residual is -1.71006e-11.
Lapse and chi remain positive. Both runs exited successfully. These two long
resolutions show improvement, but do not establish an asymptotic convergence order
or indefinite stability. The 256-cell run used the preceding scalar gauge
implementation; the shared implementation's separate 128-cell equivalence test
is documented above. The longer runs still use analytic trumpet reconstruction.

## Independent ADM gauge storage

`<adm> separate_z4c_gauge=true` allocates independent ADM lapse/shift on the first
conversion or initial halo completion. Delaying detachment preserves existing
initial-data importers which fill Z4c gauge directly. All ADM views are rebound
together. Cauchy conversion copies the lapse/shift without rescaling, including
after pre-collapsed lapse initialization, initial/regridded halo fills, and every
RK stage. ADM-to-Z4c conversion copies gauge back when storage is independent.
Restart files continue to store Z4c as the authoritative evolved state. The default
shared-storage behavior and algebraic-constraint projection schedule are unchanged.
This switch is a storage prerequisite, not a conformal evolution option.

The six new serial tests compare initial/final full-volume output for linear waves
and boosted punctures with one/two blocks, restart equivalence, and adaptive
refinement with outflow boundaries. The AMR test requires more than the eight
initial blocks and runs up to 15 cycles. Stored ADM gauge matches evolved gauge
throughout the output volume, including ghost cells. These comparisons use the
binary writer's single-precision representation. The existing 46 overhaul,
conversion and restart tests also pass. The first test run caught missing initial
halo synchronization; synchronization now occurs after the Z4c boundary fill.
No GPU/MPI or external initial-data importer execution is claimed.
The full Release executable builds and all five hyperboloidal CTests pass after
the storage changes. The new Python test passes flake8; `git diff --check` passes.
Whole-file C++ lint reports 27 pre-existing include-path/formatting findings in
the touched production files, all outside changed lines. This storage milestone
has not had a full production executable sanitizer run.

## Cartesian spherical ghost reconstruction candidate

`spherical_ghosts.hpp` now plans a spherical-domain closure on a uniform Cartesian
patch containing the entire sphere. It explicitly enumerates the nodes needed by
AthenaK's Cartesian mixed second derivatives and the wider axis-only upwind/KO
stencils. Physical nodes satisfy r<S strictly. All interpolation donors are also
strictly interior, and no ghost depends on another ghost. Required exterior nodes
are reconstructed, rather than skipping adjacent physical nodes or pretending
that every Cartesian line intersects the sphere.

For each exterior node, its true sphere-normal ray intersects interior coordinate
planes. On each plane, the planner chooses a tensor-product interpolation rectangle
containing the ray point, with all four corners inside the sphere. Convexity then
guarantees that every donor is inside. Normal-plane spacing is increased until
all required rectangles exist; a grid too coarse to support them is rejected.
The interpolated values are then extrapolated along the normal. Cartesian tensor
components use the same scalar reconstruction without rotating their components.
Applying it to reference deviations will preserve the analytic background.

This extends the smooth-data normal-line construction described in
[Baeza, Mulet and Zorío](https://doi.org/10.1007/s10915-015-0043-2)
to two transverse interpolation directions. It is not their full filtered/WENO
algorithm, nor does their analysis establish stability for Z4c.

The new CTest independently verifies donor admissibility, unique targets, mixed
corner coverage, degree-2 through degree-5 mixed polynomial reproduction, and
rejection of unsupported allocations/stencils. On an exponential field with all
nonzero Cartesian mixed derivatives, AthenaK's fourth-order Hessian operators
have boundary-region maximum error 0.0343028 at 24 cells across the diameter and
0.00720392 at 48 cells. This is a factor 4.76 improvement, not a demonstrated
fourth-order boundary Hessian result.

The evolution test is the genuinely three-dimensional, nonspherical problem
q_t + x^i partial_i q = 0 in the unit ball. Its exact solution is an off-axis
Gaussian evaluated at exp(-t)*x. The entire sphere is outflow. RK4 uses
dt<=0.15h, AthenaK's actual fourth-order `Lx` upwind operator, and KO6 with
coefficient 0.1. At t=1.5:

| Normal degree | N coarse/fine | RMS error coarse/fine | Maximum error over time coarse/fine |
|---|---|---|---|
| 3 | 24 / 48 | 6.92734e-4 / 5.56929e-5 | 0.0786323 / 0.00912019 |
| 4 | 24 / 48 | 6.82070e-4 / 5.49525e-5 | 0.275137 / 0.00985624 |
| 5 | 32 / 64 | 2.30492e-4 / 1.99636e-5 | 0.202205 / 0.00500660 |

The quartic 48-cell run reaches t=8 with final RMS error 8.00790e-7 and maximum
error over all steps 0.00985624. Both final and time-maximum errors must decrease
in the convergence test. Large coarse-grid transients remain visible in the table.

Several alternatives were rejected. Direct lattice-line extrapolation, selecting
the best-aligned lattice line, and blending lattice lines exceeded the initial
error cutoff. With the final normal-plane reconstruction but **centered**
advection, the maximum error exceeds 1000 by t=0.5125 at N=24 and t=0.2875 at N=48.
Thus geometric consistency alone is insufficient. The negative control can be
reproduced with `hyperboloidal_ghost_tests 48 4 1.5 0.1 centered`; it must fail.
The corresponding upwind case omits the last argument.

**This is still a boundary candidate, not a completed hyperboloidal Z4c boundary.**
The conformal-wave tests below now expose additional limitations; nonlinear
Cartesian Z4c has still not been evolved with this closure.
The conformal mesh adapter now provides `AddMeshUpwindAdvection`, replacing only
the componentwise beta-gradient advection by AthenaK's `Lx`, while retaining
centered geometric derivatives. Independent coefficient-table tests check every
packed field, both shift signs, and second/fourth/sixth-order operators; the CMC
Minkowski fixed point remains stationary. This adapter is not yet dispatched in
a Cartesian conformal evolution. Scri regularity, gauge
characteristics, stiffness near arbitrarily small positive Omega, runtime
dispatch, and multiblock/AMR donor mapping remain unresolved. No stable production
scri switch is enabled by these scalar transport results.

Validation for this milestone: all six Release CTests pass, with the mesh adapter
rerun after adding the upwind correction. The adapter also passes a strict
ASan/UBSan build and execution. Changed C++ files pass lint and the diff passes
whitespace checks. The full ghost/transport sanitizer process was still running
when this milestone was recorded; it is not counted as a completed sanitizer pass.

## Conformal-wave boundary audit and interior dissipation

The Cartesian boundary candidate is now tested on a nonspherical conformally
invariant scalar wave, not only one-way transport. With S=a=1 and Pi=n_bar(phi),
the test evolves

```
phi_t = beta.grad(phi) + alpha Pi
Pi_t = beta.grad(Pi) + alpha Lap(phi) + grad(alpha).grad(phi)
       - 3 Pi - alpha R phi/6
alpha=(1+r^2)/2, beta=-x, R=6/alpha^3-6/alpha.
```

The exact dipole is a physical Cartesian derivative of
`[F(T-Rphysical)-F(T+Rphysical)]/Rphysical`, divided by Omega, with
`F(s)=0.01 exp(-((s-0.6)/0.2)^2)`. Its axis is (0.3,0.4,sqrt(0.75)).
`check_wave_exact.py` independently differentiates this expression using
45-digit arithmetic and checks both normal-momentum and coordinate-time-momentum
PDE forms at twelve spacetime points; maximum residual is 2.80260e-44.

The initial quartic closure looked convergent at t=1.2: N=24/48/96 RMS errors were
0.0600233/0.00231783/0.000478150. However, its N=48 error grew to 0.364675 by t=4.
Increasing ordinary ghost-filled KO from 0.1 to 0.5 made the maximum error exceed
1000 by t=1.70313. Quintic extrapolation failed that cutoff by t=1.94375. Switching
to the equivalent coordinate-time-momentum equation also failed by t=3.35312.
Short-time convergence therefore did not establish a usable long-time closure.

Cubic extrapolation improved the normal-momentum wave. With ordinary KO, N=48/96
runs at t=4 give RMS errors 2.71609e-4/6.11523e-6 and maximum errors over time
0.112053/0.00700844. These are about 44x and 16x improvements under refinement.

`interior_dissipation.hpp` adds a further alternative. It uses
`Q=-sum_d D3_d^T D3_d/(64 h_d)`, retaining only four-point lines whose entire
stencil is active. Consequently `sum q Qq=-sum_lines (D3 q)^2/(64 h_d)<=0`
in the uniform-grid squared norm. No extrapolated donor enters this operator.
An independent row-wise test verifies the identity, zero total Q, annihilation
of quadratics, and untouched inactive nodes, including an interior hole filled
with NaN sentinels. It equals KO6 in the interior but has only second-order
boundary consistency. The scalar identity is not a Z4c energy estimate.

With cubic extrapolation and this operator at coefficient 0.1, the final-source
wave regression measures:

| N | End time | RMS field error | Maximum field error over all steps |
|---|---|---|---|
| 24 | 1.2 | 5.00945e-3 | 1.23487 |
| 48 | 1.2 | 1.86630e-4 | 0.0978664 |
| 48 | 4 | 2.49246e-4 | 0.0978664 |
| 48 | 8 | 3.63415e-4 | 0.0978664 |

The coarse transient is substantial. Optional CSV histories retain unweighted
field errors and fixed-background time-translation energy,
`E=integral[alpha(Pi^2+|grad phi|^2+R phi^2/6)/2+Pi beta.grad(phi)] d^3x`,
alongside energy computed from the exact solution on the same active nodes.
For the N=48 run, numerical energy starts at 0.196317, peaks at 0.196584, and is
3.97304e-9 at t=4. Energy degenerates at scri, so its smallness cannot replace
the unweighted error checks. Late residual field error still rises slightly;
at t=8 its energy is 8.14910e-9 and instantaneous maximum field error is
0.00329551. Indefinite stability is not established. Raising the interior-only
coefficient to 1 reduces the early maximum slightly but increases t=8 RMS error
to 4.91112e-4, so it is not selected as an improvement. Quartic extrapolation with
that stronger interior operator still has t=4 RMS error 0.0179.

Reproduction: `hyperboloidal_wave_tests N degree end [dissipation [form [history.csv]]]`.
Forms are `normal`, `coordinate`, `normal_ko`, and `coordinate_ko`; the `_ko`
forms use interior-only dissipation. The default CTest checks cubic `_ko`
convergence and the t=4 field/energy bounds. Failed alternatives remain selectable.

All eight Release CTests pass. The new dissipation unit test and short normal/
coordinate wave runs also pass strict ASan/UBSan builds and execution. These
smokes are not full-duration wave sanitizer tests. The previous extended transport
sanitizer process is still being tracked separately. Changed C++ and Python files
pass lint. No production Z4c runtime dispatch or Cartesian puncture evolution is
claimed by this wave milestone.


## Nonlinear Cartesian conformal patch

`cartesian_patch.hpp` connects the tensor conformal RHS, shared live gauge,
actual AthenaK field layout and fourth-order Cartesian derivatives to the
cubic true-normal spherical ghost reconstruction. The active domain is r<S,
inside one uniform Cartesian allocation. The constructor checks stencil
coverage, so an invalid/coarse geometry is rejected. Every RHS reconstructs
reference deviations from strictly interior donors, restores analytic reference
jets and replaces only advection by AthenaK's upwind operator. The advecting
velocity is the full evolved shift, not its deviation. Interior-only KO has
coefficient 0.1. Auxiliary B fields remain frozen for the integrated shift gauge.

The only RHS subtracted is the analytic CMC Minkowski floating-point residual;
no evolved or black-hole RHS is subtracted. Omega is never floored. Invalid
spatial metrics/lapses, aliased input/output or scratch storage, invalid damping
coefficients and nonfinite RHS values are rejected. The adapter writes zero RHS
to inactive nodes. It does not dispatch native AthenaK tasks, convert physical
ADM fields, perform algebraic projection, or evolve punctures yet.

Diagnostics include unweighted active-node RMS H, conformal norms of the momentum
covector and Z4 covector, physical Theta, determinant and trace-free residuals,
minimum chi/lapse and maximum field deviation. In an outer shell of width twice
the largest grid spacing, they also record the maximum absolute conformal pole
numerator, its deviation from the exact CMC numerator, and the null residual's
deviation from CMC. The raw numerator need not vanish at finite Omega even for
exact CMC. Shell measurements are not evaluations at scri and do not prove that
the pole/Omega limit exists.

The new executable uses SSPRK3 and dt=min(0.025h,0.04 min(Omega)), with the last
step shortened to the requested end time. This conservative empirical step
bound is for the tested S=a=1 and kappa1=5; it is not a general characteristic
CFL theorem. Each simulation uses Kokkos Serial on one CPU core.

Tests include poisoned inactive cells (no contamination of active RHS or
constraints), rejection of a positive-determinant but indefinite metric, alias
rejection, and a continuum constraint check independent of finite differences:
the instantaneous H derivative for an analytic lapse perturbation of flat CMC
initial data is zero to 2.3e-14. Unperturbed Cartesian CMC remains stationary to
1.4e-15 at t=0.01 on N=24.

Two nonspherical lapse-only pulses of amplitude 1e-4 multiply
(1+0.2x+0.3yz). The compact pulse is exp(1-1/(1-r^2/0.36)) inside r<0.6,
zero outside. Its short-run Hamiltonian convergence is irregular, so a smoother
pulse, (1-r^2)^4 exp(-r^2/0.25), tests refinement without that narrow transition.
Both start with exactly constraint-satisfying spatial Minkowski data.

| Profile | N | t | RMS H | RMS M | RMS Z |
|---|---|---|---|---|---|
| Compact | 24 | 0.01 | 4.12474e-6 | 3.97262e-5 | 7.13782e-6 |
| Compact | 36 | 0.01 | 2.00999e-6 | 1.62353e-5 | 2.00531e-6 |
| Compact | 48 | 0.01 | 2.01377e-6 | 1.02932e-5 | 6.13443e-7 |
| Smooth | 24 | 0.01 | 8.34771e-7 | 5.63086e-6 | 7.99096e-7 |
| Smooth | 36 | 0.01 | 7.29736e-8 | 1.28517e-6 | 2.21033e-7 |
| Smooth | 48 | 0.01 | 3.62252e-8 | 6.99048e-7 | 1.25695e-7 |
| Compact | 24 | 0.5 | 8.01329e-6 | 5.29683e-5 | 1.30449e-5 |

The smooth pulse improves on all three grids, but these results do not establish
uniform fourth-order convergence. The N=24 compact t=0.5 run completes 1754 steps
with minimum lapse 0.502778, minimum chi 1.00000, and maximum field deviation
2.91790e-4. Long-time stability and fine-grid evolution through a full crossing
time remain unverified. The N=24 smooth t=2 extension subsequently completed: H=1.87915e-5,
M=1.06217e-4, Z=1.72132e-5, minimum lapse=0.502881, minimum chi=0.999980,
and maximum field deviation=2.65935e-4 after 7014 steps. This is still a
single coarse resolution and does not establish long-time convergence.

Reproduce with `hyperboloidal_cartesian_tests N end amplitude [compact|smooth]`.
The default test checks CMC stationarity, the compact pulse and a 24/36 smooth
constraint-convergence pair. All nine Release CTests pass; after final diagnostic
changes the Cartesian CTest passes again. The normal AthenaK executable builds.
Strict ASan/UBSan builds and one-step reference/pulse smokes pass (not the full
Cartesian evolution durations). Changed C++ files pass cpplint.

The previously pending full spherical ghost/transport ASan/UBSan run also
completed with exit 0, including cubic/quartic/quintic convergence pairs and the
N=48 quartic t=8 case. That binary preceded only the centered-advection CLI
negative control; a rebuilt final-source short upwind smoke also passes. These
results close that earlier pending validation, not the nonlinear stability gap.


## Cartesian trumpet initialization and first puncture runs

`cartesian_radial.hpp` converts spherical scalar, vector and tensor jets from
(r,0,0) to arbitrary directions, transforming all derivative indices, including
mixed Cartesian second derivatives. Finite differences of independently sampled
component values converge by about four on each halving of their step, for both
a manufactured nonflat radial metric and CMC trumpet data. Off-axis initial
Hamiltonian and momentum constraints over masses 0.05/0.5, radii 0.02/0.15/0.6/0.95
and three directions are below 2.7e-13.

`cartesian_trumpet.hpp` initializes the actual AthenaK field arrays from these
jets. It requires S=a=1 and a grid excluding r=0 and scri; it does not floor either
singular location. An optional fixed initial-profile reconstruction differentiates
only changes from that profile and adds its exact jets back. Ghost deviations
still use strictly interior donors and the true-normal spherical plan. The
profile is not the gauge target: Minkowski remains the reference. No black-hole
RHS is subtracted. The test explicitly checks that the initial geometric RHS is
stationary to 1e-6 while the live gauge RHS is nonzero; at N=24, M=0.5 their maxima
are 9.94e-12 and 0.510341 respectively.

The patch now also has a masked final-step algebraic projection, matching the
existing AthenaK normalization of det(g_tilde) and removal of tr(A_tilde), with
rejection of invalid metrics instead of a determinant floor. Tests verify that
it preserves inactive poisoned cells and rejects indefinite metrics. It does
not project Hamiltonian, momentum or Z4 differential constraints.

Run `hyperboloidal_cartesian_tests N end mass trumpet` for projected evolution,
or use `trumpet_raw` for the unprojected control. These are experimental Cartesian
puncture runs, still outside AthenaK's production task graph. N=24, M=0.5 runs
to t=0.1 with positive lapse and chi, but the errors are not yet satisfactory:

| Treatment | RMS H | RMS M | RMS Z | max det error | max trace error |
|---|---|---|---|---|---|
| No projection | 0.0208668 | 0.0939474 | 0.0289368 | 2.08e-4 | 2.89e-3 |
| Projection | 0.0208484 | 0.0939028 | 0.0289517 | 8.88e-16 | 1.22e-15 |

The maximum projected H error is 0.276497 at r=0.929108; maximum M is 0.923621
at r=0.992846. This localizes the serious differential-constraint error to the
outer region and shows that algebraic projection is not a solution to it. The N=36 projected comparison at t=0.1 gives H=0.00184574, M=0.0127911,
Z=0.00355659: improvements of about 11.3, 7.34 and 8.14 over N=24. This is
encouraging refinement evidence, not an established asymptotic order. The N=24
run completes t=1 with positive lapse/chi, but H=0.345206, M=1.45786 and
Z=0.424791 are too large to claim an accurate long evolution. No converged or
long-time-stable Cartesian black-hole evolution is claimed. The next boundary
audit must address the singular pole assembly/regularity near arbitrary Cartesian
cuts, rather than interpreting the scalar boundary successes as sufficient.

All ten Release CTests pass, including the new independent radial derivative
and constraint tests and the one-step puncture smoke. The final localization
run reproduces the t=0.1 constraints. Strict ASan/UBSan radial tests and a
one-step Cartesian trumpet smoke pass. Full-duration Cartesian puncture
sanitizer tests and native task integration remain outstanding.


The optional final CLI argument selects the pole timestep coefficient (default
0.04, allowed experimental range (0,0.2]). At N=24, M=0.5, t=0.1, increasing it
to 0.1 reduces the number of steps from 351 to 141. H changes from 0.02084844 to
0.02084788, M from 0.09390278 to 0.09390201, and Z from 0.02895168 to 0.02895091.
Those changes are much smaller than the spatial-refinement changes. This is an
early-time timestep sensitivity check, not proof that the larger step is stable
at later times. Finer and longer runs with that coefficient are tracked separately.


## Native AthenaK hyperboloidal runtime

The standard executable now supports `z4c/hyperboloidal=true` with
`problem/pgen_name=z4c_hyperboloidal`. The example
`inputs/z4c/hyperboloidal.athinput` runs a bounded two-step CMC smoke. Set
`problem/mass=0.5` for the trumpet, or `problem/lapse_pulse=1e-4` for a nonspherical
smooth lapse pulse. S=a=1, physical K_ref=-3, and the shared live gauge coefficients
remain (2,0.1,1.5,1). Hyperboloidal damping/dissipation have separate parameters,
`hyperboloidal_kappa1=5` and `hyperboloidal_dissipation=0.1`. No Cauchy chi floor
is permitted. Initialization rejects unsupported geometry/options, rather than
silently reusing Cauchy evolution.

Supported scope is one uniform three-dimensional MeshBlock on one rank, three
halo cells, fourth-order spatial derivatives, chi power -4 and SSPRK3 (`rk3`).
The sphere r<1 must fit inside the physical mesh. Matter, multilevel meshes,
trackers, horizons and wave/CCE extraction are rejected for this prototype.
This is a runtime integration milestone, not a claim that the long-time numerical
stability problem has been solved.

The native task chain now calls the Cartesian conformal RHS, reconstructs the
spherical ghost fringe, performs masked final-stage algebraic projection and
converts physical ADM at every RK stage. Cartesian-box Sommerfeld/outflow
operations do not overwrite the spherical closure. The physical timestep is
min(0.025 min(dx), hyperboloidal_pole_cfl min(Omega)), with default pole coefficient
0.04; the module cancels the framework's additional CFL multiplier so this agrees
with the test driver. The mesh still shortens the final step to the requested
end time. The empirical coefficient is not a general nonlinear CFL theorem.

ADM lapse storage is detached automatically before writing alpha_phys=alpha_bar/
Omega. Physical metric, curvature and psi4 are written only to active cells;
inactive ADM entries are NaN and are accompanied by `z4c_active` in field outputs.
The evolved conformal variables remain authoritative. Every field output,
including a single selected scalar, carries the mask. Unsupported derived
curvature, Weyl and PDF outputs are rejected because their derivative/reduction
paths are not yet mask aware.

Native constraint output contains physical H and the momentum covector; `con_M`
and `con_Z` are their **conformal** squared norms, avoiding artificial suppression
near scri. `con_C` is H^2+M_conformal^2+Z_conformal^2+Theta^2. Inactive entries are
zero and identified by the mask. History reports unweighted active-node RMS
constraints, algebraic residuals, positive-field minima and outer-shell pole/null
diagnostics. It neither integrates the divergent physical volume at scri nor
excises difficult cells based on chi.

A restart reconstructs the immutable analytic initial profile before loading the
checkpoint's evolved fields. It then refreshes ADM and constraints without
reinitializing the evolution. This avoids incorrectly pairing restart values
with initial-data derivatives.

Validation: the initial native/Cauchy regression run passed 61 tests (nine new
native checks plus 52 existing overhaul, conversion, restart and separate-gauge
checks). Four additional output-safety/boundary checks bring the native test file to 13
passing tests; the final combined Release run passes all 65 tests. A grid with
dyadic spacing places nodes exactly at r=1 and verifies they remain inactive,
with NaN physical ADM values and finite, stationary active CMC data. All 13 native
tests also pass under ASan/UBSan (the initial 12 plus the final exact-scri test). These check CMC, a nonspherical pulse and a trumpet against the
independent driver, physical ADM mapping at the initial and final saved states, poisoned inactive
ADM cells, restart equality, unsupported configurations and output masks.
A native N=24, M=0.5 evolution reaches t=0.1 in 141 steps with pole coefficient
0.1, giving H=0.0208478807307, M=0.0939020062887 and Z=0.0289509126457, matching
the standalone run. Its large errors are reported, not treated as an accuracy pass.

Additional standalone refinement evidence: at N=48, t=0.1 and pole coefficient
0.1, H=4.07348e-4, M=2.75813e-3 and Z=6.27472e-4. Errors decrease further from
N=36. The largest errors remain near r=0.977. The N=36 run reaches t=0.5 with
H=0.0177828, M=0.109239 and Z=0.0282965, substantially below N=24 at that time
but still too large to establish accurate long-time evolution. Stability and
asymptotic convergence across longer native puncture runs remain outstanding.

### Controlled boundary-order comparisons

The native input `z4c/hyperboloidal_ghost_degree` and the standalone driver's
last optional argument select polynomial degree 2 through 5. The default remains
3. This changes the true-normal interpolation/extrapolation degree only: the
required stencil halo stays at three cells, and every donor remains strictly
inside the sphere. There is no fallback to exterior donors or a shorter halo.
The N=24 example cannot accommodate degree 5 and rejects it with
`no interior normal-ray rectangles`; N=36 accommodates it. These are geometric
requirements, separate from numerical stability.

The following standalone comparisons use exactly the same N=24 grid, mass 0.5,
projected SSPRK3, pole coefficient 0.1, dissipation 0.1 and live reference gauge.
All reach t=0.5 in 702 steps. H, M and Z are unweighted active-node RMS, with
conformal norms for the latter two:

| Ghost degree | H | M | Z | Maximum shell null deviation |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.0430851130 | 0.198479468 | 0.0601857403 | 0.0834984 |
| 3 | 0.101036777 | 0.679817958 | 0.172531074 | 0.124565 |
| 4 | 0.579389655 | 2.26506907 | 0.542547835 | 0.429136 |

Higher interpolation order is clearly not sufficient to improve this coarse
nonlinear evolution. Degree 2 reduces these errors, but does not solve the
long-time problem: its N=24 continuation reaches t=1 in 1403 steps with
H=0.210158449, M=0.821378051 and Z=0.171307931. Lapse and chi stay positive,
but the growing constraint errors are unacceptable as an accuracy result.
The refined degree-2 run and the native N=48 cubic run are still pending;
neither is counted as a completed validation here.

The boundary policy is an experimental extrapolation closure, not a proven
constraint-preserving characteristic boundary condition at scri. Short native
comparisons, restart equality and memory-safety tests do not establish a
nonlinear energy estimate or accurate long-time puncture evolution.

Validation for the selectable-degree change: the Release build passes 75 tests
(23 native plus 52 existing Cauchy/ADM/restart checks). The native cases compare
CMC, a nonspherical pulse and a trumpet with the standalone driver for degrees
2, 3 and 4; degree 5 is checked on N=36, alongside the explicit N=24 geometric
rejection. Restart equality is checked for degrees 2 and 3. All 23 native tests
also pass with ASan/UBSan. Seven focused CTests pass: reference building blocks,
constraints, tensor RHS, mesh adapter, interior dissipation, Cartesian evolution
and Cartesian radial jets. C++ and Python lint and diff whitespace checks pass.

### Further boundary trials and single-core cost

The degree-2 N=36, M=0.5 run reaches t=0.5 in 2607 steps with
H=0.0137708900, M=0.0930641297, Z=0.0175467266 and shell null deviation
0.0631063. Relative to N=24, these constraint norms decrease by factors
3.13, 2.13 and 3.43. This is encouraging refinement evidence, but the
improvement over cubic interpolation is smaller on N=36 than on N=24;
no asymptotic convergence order or long-time stability is inferred.

Two native degree-2 N=24 trials increase interior-only KO dissipation while
holding mass, timestep coefficient, gauge and end time fixed (M=0.5,
pole coefficient 0.1, t=0.5):

| Dissipation | H | M | Z |
| --- | ---: | ---: | ---: |
| 0.1 (baseline) | 0.0430851130 | 0.198479468 | 0.0601857403 |
| 0.5 | 0.0486755141 | 0.200334942 | 0.0595339257 |
| 1.0 | 0.0596232163 | 0.204863964 | 0.0587677701 |

Increasing dissipation is not a remedy for this error growth.
A separate, unmerged driver experiment extrapolated the entire RHS from
interior donors across an outer layer of fixed width in grid cells, with
all constraint diagnostics still evaluated on the entire original active
sphere. Quadratic true-normal stencils with verified donor/target coverage
were used. At the same N=24 and t=0.5, a one-cell layer gives
H=1.34331414, M=1.42395855, Z=0.720511753 and null deviation 2.36473;
a half-cell layer gives H=0.0446443112, M=0.0829081576, Z=0.0976181607
and null deviation 0.0901278. The latter reduces momentum error but increases
Z error. These mixed/poor results do not justify a native runtime option.
The experiment is excluded from the solver. No singular denominator is floored.

A three-second sample of the native N=48 run found substantial cost in the
interior dissipation kernel (345 of 2312 samples). `CartesianComponent` now
uses direct field-relative access when all three spatial strides are contiguous,
with a fallback using the actual strides for padded storage. The parent view
remains owned by the calling patch while kernels execute. No floating-point
stencil operation or boundary condition changes.
An independent read/write oracle checks both contiguous and deliberately padded
five-dimensional views, including non-unit innermost stride and field padding.
A single before/after N=24, t=0.05, degree-2 puncture benchmark takes 7.64 versus
5.46 CPU seconds; all 26 common printed fields match exactly. This timing is
machine/load dependent, not a portable speedup guarantee.
The native N=24 cubic t=0.1 repetition matches the pre-change final history row
and every saved Z4c, ADM and constraint field exactly, including inactive values.

Validation of the addressing optimization: all 75 Release native/Cauchy
regressions and all 23 native ASan/UBSan tests pass. The Cartesian evolution
CTest passes, including the layout oracle and smooth-pulse refinement check.
The strict standalone ASan/UBSan build with warnings-as-errors passes a short
CMC evolution, poisoned-inactive-cell checks and both layout oracles. C++ lint
and diff whitespace checks pass. Longer N=36/N=48 native degree-2 runs to t=1
are pending and are not included among the passed checks.

### Radial constraint budgets

`tst/hyperboloidal/analyze_native_constraints.py` reads masked native constraint
binary dumps and emits JSON containing unweighted global RMS/maxima, the
coordinates of each maximum, and radial-bin RMS and fractions of each squared
constraint norm. M and Z use the conformal norms already stored by the native
solver. Every active cell must be covered; incomplete radial bins, inconsistent
masks and nonfinite/negative squared constraint values are rejected. The current
prototype supports one uniform block and the unit spherical domain.

For example, `python tst/hyperboloidal/analyze_native_constraints.py
path/to/hyp.con.00001.bin` uses bins with edges
0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95 and 1. Custom edges are supplied with
`--edges`. These are compactified coordinate radii, not physical areal radii.
The fractions partition the *squared* global norm; they are not percentages of
pointwise error or physical volume integrals.

At N=24, M=0.5, t=0.5, KO coefficient 0.1 and pole coefficient 0.1:

| Ghost degree | H squared norm at r>0.9 | M squared norm at r>0.9 | Z squared norm at r>0.9 |
| --- | ---: | ---: | ---: |
| 2 | 37.2% | 85.4% | 95.8% |
| 3 | 66.9% | 96.5% | 97.8% |

The H maximum lies at r=0.929108; the momentum maximum lies at r=0.992846.
The stronger-KO trials have a distinct additional problem: the fraction of the
H squared norm at r<0.25 increases from 4.96% (KO=0.1) to 23.3% (KO=0.5) and
46.5% (KO=1). Increasing dissipation does not merely suppress an outer error;
it introduces substantial additional error in the puncture region.
The budget reader agrees with native global history norms to binary-output
precision for CMC, nonspherical pulses and trumpets at degrees 2, 3 and 4,
and on the dyadic grid containing exact scri nodes. Tests verify complete cell
and squared-norm accounting and rejection of bins omitting the interior.
All 23 native integration tests pass with these additional budget checks.

A further unmerged experiment separates the normal and transverse polynomial
degrees in the sphere-normal ghost construction. Interior-donor checks and
mixed polynomial reproduction pass (errors below 5e-15 for the tested pairs).
At the same N=24, t=0.5, normal degree 2/transverse degree 3 gives
H=0.0439600743, M=0.202703694 and Z=0.0604731641; normal degree 3/transverse
degree 2 gives H=0.105457353, M=0.682144538 and Z=0.173264921. Neither improves
the corresponding equal-degree baseline. This option is not adopted.

The native N=36 quadratic snapshot at t=0.500127604 (the first output step
past 0.5) gives H=0.0137806632, M=0.0931146941 and Z=0.0175563926.
The fractions outside r=0.9 are 70.9%, 95.7% and 99.0%, respectively.
This supports localization of the remaining errors near the outer boundary;
it is not a same-time field self-convergence comparison with the t=0.5 dump.

Another unmerged trial reconstructs the physical Theta ghost values as Omega
times an extrapolation of Theta/Omega. It preserves a manufactured
Omega*(1+0.3*x*y+0.4*z*z) field to below 4e-15 on the tested stencils.
At N=24, t=0.5, quadratic ghosts give H=0.0431642946, M=0.197438132 and
Z=0.0611846130; cubic ghosts give H=0.100823596, M=0.536190840 and
Z=0.164510993. The cubic momentum norm improves, but this is not a uniform
improvement across constraints and policies. A matched N=36 cubic comparison
at t=0.1 (both pole coefficient 0.1, 522 steps) gives:

| Theta ghosts | H | M | Z |
| --- | ---: | ---: | ---: |
| Direct baseline | 0.00184573849 | 0.0127910624 | 0.00355657823 |
| Factored | 0.00185588174 | 0.0114052407 | 0.00356956622 |

The finer result again trades lower momentum error for slightly higher H and Z.
Factoring only this ghost variable is not adopted as a stability fix or exposed
in the native input. The production boundary treatment remains unchanged.

### Interior Hawking-mass diagnostic

`z4c/hyperboloidal_mass_diagnostics=true` appends Hawking masses and physical
areal radii on coordinate spheres r=0.3, 0.5 and 0.7 to native history. The
columns are `mH-r03`, `Rarea-r03`, `mH-r05`, `Rarea-r05`, `mH-r07` and
`Rarea-r07`. The flag defaults to false. `hyperboloidal_mass_nmu` defaults to 32
and selects Gauss-Legendre nodes in cos(theta), with twice as many equally
spaced azimuthal points; allowed values are 4 through 128. Increasing this
parameter controls angular quadrature error, not Cartesian interpolation error.

The definition is the full 3D surface integral

```
m_H = sqrt(A/(16*pi)) * [1 + integral(theta_+ theta_- dA)/(16*pi)],
```

with null normals n+s and n-s, whose inner product is -2; see equation (4.8) of
[Csukás and Rácz, Hyperboloidal initial data without logarithmic singularities](https://doi.org/10.1007/s10714-025-03424-y).
No spherical symmetry is assumed when evaluating the evolved fields. In a
spherically symmetric Schwarzschild continuum solution this mass is constant,
but that statement does not hold for arbitrary surfaces in a general spacetime.
This finite-radius diagnostic is neither an apparent-horizon finder nor a
Bondi-mass measurement at scri.

For b_ij=gtilde_ij/chi, B=b^ij x_i x_j and sbar^i=b^ij x_j/sqrt(B), the surface
mean curvature in the physical spatial metric is
`H_s = Omega div_b(sbar) - 2 sbar^i partial_i Omega`. The null expansion product
is `(K_ss-K)^2-H_s^2`. The code constructs K_ss-K from the full physical ADM
curvature, including any numerical trace of Atilde, rather than assuming that
this algebraic constraint vanishes. The area density per Euclidean solid angle
is `J=r*sqrt(det(b)*B)/Omega^2`. For improved summation of nearly cancelling
terms, the code integrates `J*theta_+*theta_- + 4`: the added constant has exact
sphere integral 16*pi. It does not subtract the initial black-hole mass or force
an evolved value to remain constant.

The densities are computed from reconstructed mesh jets and interpolated onto
each surface with tricubic stencils. All 64 donors must be strictly inside the
active sphere; otherwise extraction is rejected. Inactive cells remain poisoned.
The interpolation bias must be measured independently of evolution error. On
exact M=0.5 trumpet initial data, the maximum mass error over the three extraction
spheres decreases from 4.83e-3 at N=24 to 8.55e-4 at N=36 and 2.30e-4 at N=48
(the largest error is at r=0.7). These numbers include the default 32-node
quadrature. Doubling its angular resolution changes each tested mass by less
than 15% of its Cartesian interpolation error. The original 24-node choice
failed that check at N=48 and is not the default.

Independent validation includes analytic Minkowski and Schwarzschild CMC
spheres (M=0.05 and 0.5, radii 0.2 through 0.9), nonzero physical Theta, and a
nonzero A trace. A finite-difference oracle in sheared flat coordinates verifies
the surface mean curvature with second-order error reduction. Nonlinear flat
coordinate transformations test nonzero off-diagonal metric derivatives and
known physical area. Native enabled/disabled comparisons show identical saved
Z4c, ADM and constraint fields, and mass-enabled restart histories agree with
uninterrupted ones. The full Release regression run passes 79 tests. All 27
native ASan/UBSan tests passed; after retaining the small A-trace residual in the
physical curvature contraction, all six affected native tests passed again in
both Release and ASan/UBSan. The final standalone Hawking CTest also passes in Release and in the strict
ASan/UBSan build with warnings treated as errors.

To add these diagnostics to an older checkpoint, use a minimal input overlay
with `-r checkpoint.rst -i mass_only.athinput`; a command-line override alone
cannot add parameter names absent from the checkpoint. The overlay need contain
only the new z4c diagnostic parameters. Setting `time/nlim=0` in that overlay
permits a diagnostic-only restart without advancing the solution. The checkpoint
audits below used this path and report zero MeshBlock-cycles.

Completed native evolution evidence now includes the cubic N=48, M=0.5 run to
t=0.5 (3076 steps), with H=0.00855585661, M=0.0642931265 and Z=0.0128601265.
These are lower than the cubic N=36 results at the same time, but do not by
themselves establish long-time stability. The quadratic N=36 run reaches t=1
(5213 steps), with H=0.0372859885, M=0.165782053 and Z=0.0417192134. A quadratic
N=24 continuation, restarted at t=0.5, reaches t=1 with H=0.210158322,
M=0.821378419 and Z=0.171308161. Its extra shortened step at the restart explains
small differences from the uninterrupted standalone result.

Measured masses on the saved evolved data are:

| Policy / N / time | m_H at r=0.3 | m_H at r=0.5 | m_H at r=0.7 |
| --- | ---: | ---: | ---: |
| Quadratic / 24 / 0.5 | 0.49966937 | 0.50019175 | 0.49490599 |
| Cubic / 24 / 0.5 | 0.49966941 | 0.50015816 | 0.48846297 |
| Quadratic / 36 / 0.500127604 | 0.49992726 | 0.50032228 | 0.49732945 |
| Cubic / 48 / 0.5 | 0.49997666 | 0.50015303 | 0.49924718 |
| Quadratic / 24 / 1 | 0.50090156 | 0.49528493 | 0.53876358 |
| Quadratic / 36 / 1 | 0.50011581 | 0.49868029 | 0.51186014 |

The expected continuum mass is 0.5. Differences include both evolution and
extraction interpolation errors. At N=36, t=1, doubling the angular quadrature
changes the outer mass by 5.61e-5; at cubic N=48, t=0.5 the change is 4.21e-6.
Thus angular quadrature is not the principal source of these outer deviations.
The N=48 quadratic t=1 run remains pending. The coarse quadratic N=24
continuation requested to t=5 instead fails an active physical-ADM validity
check after the last saved history at t=1.300126953. At that output H=0.452209,
M=5.22193 and Z=1.31884, with rapidly growing outer pole residuals. This is a
failed evolution, not a stability pass. A replay with mass diagnostics disabled
fails after its last logged cycle at t=1.367852; its shared history columns are
bitwise identical to the enabled run at t=1.100517578, 1.200322266 and 1.300126953.
A second replay reducing the pole timestep coefficient from 0.1 to 0.04 fails
after its last logged cycle at t=1.375266. The smaller timestep is therefore not
a cure, and mass extraction is not necessary for this failure. At the final
saved snapshots near t=1.350, the smallest conformal-metric eigenvalue is
0.103888 (coefficient 0.1) or 0.106223 (0.04), at
(x,y,z)=(0.04375,0.65625,0.74375), r=0.992846. The metrics are still positive
definite at these saved times but strongly distorted near scri. The exact
invalid field at the subsequent failing stage has not yet been isolated.
