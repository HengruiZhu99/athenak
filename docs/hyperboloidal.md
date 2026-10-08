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

**The requested production 3D solver is not complete.** The AthenaK Z4c task
graph and ADM conversion are unchanged. The CMake option builds the tests and
standalone spherical executable; it does not enable a hyperboloidal AthenaK
runtime mode. Cartesian spherical-boundary stencils, full mesh integration and
longer puncture stability tests remain outstanding. GPU/MPI testing is out of
scope. The sections below retain the equation derivations and earlier failed
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

A concrete integration obstacle is that `coordinates/adm.cpp` currently aliases
ADM lapse to Z4c lapse. The physical lapse and evolved Penrose lapse differ by
Omega, so a conformal runtime must allocate independent ADM lapse storage before
using this conversion. The adapter intentionally returns point values rather than
writing physical lapse through the current shared alias. The production task graph
remains unchanged and no incomplete hyperboloidal runtime switch has been exposed.

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

The next integration steps remain independent physical ADM gauge storage, a
validated sphere-crossing stencil/closure and active-cell policy, and dispatch of
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
and Python regression scripts pass lint. A finer 512-cell t=20 run is being tracked
separately; its incomplete history is not counted as a passed long-time test here.
