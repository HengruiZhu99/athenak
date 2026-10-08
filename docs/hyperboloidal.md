# Experimental hyperboloidal building blocks

## Status

**This branch does not implement a complete hyperboloidal Z4c evolution.** It adds
regular CMC reference geometry, a factored reference gauge kernel, and a stable
boundary-fitted radial characteristic prototype. It also includes off-shell physical
Hamiltonian/momentum and Z4 constraint kernels in regular conformal variables.
The nonlinear conformal interior tensor RHS is implemented as an experimental
kernel, with its scri pole numerators exposed. The production Z4c RHS, mesh task
graph, and ADM conversion are unchanged. There is no hyperboloidal runtime mode.
The CMake option only builds tests.

This is a partial implementation of the requested 3D hyperboloidal solver. A stable
model transport problem, an exact reference solution, and a stationary gauge do not
establish off-background regularity or stability of conformal Z4c. In particular,
the reference Einstein sources below must not be used as the sources for a perturbed
metric. The unresolved equations and integration work are listed below.

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
