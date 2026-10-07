# Experimental hyperboloidal building blocks

## Status

**This branch does not implement a complete hyperboloidal Z4c evolution.** It adds
regular CMC reference geometry, a factored reference gauge kernel, and a stable
boundary-fitted radial characteristic prototype. The production Z4c RHS, mesh task
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

1. Derive and implement the full off-constraint conformal Z4c equations, including
   definitions of Theta and Khat, connection constraints and damping. Reference
   Einstein sources cannot replace the metric-dependent Omega^-1/Omega^-2 terms.
   Prove the limiting combinations at scri and verify them on nontrivial data.
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
