# Algebraic-constraint audit and isolated projection helper

The original drift audit concerns **existing local CPU puncture snapshots**.
A separately compiled MPI candidate now connects the helper to accepted RK
steps, as described below. It changes no running/frozen Aurora configuration
and supplies no evidence of long-time stability or convergence.

The spherical donor reconstructs `g_tt = 1/sqrt(g_rr)` and
`A_tt = -A_rr*g_tt/(2*g_rr)` on every evaluation
([spherical_tensor.hpp](../../src/z4c/hyperboloidal/spherical_tensor.hpp)). Thus
unit determinant and traceless curvature hold algebraically in the 1D chart.
The composite engine evolves all six metric and curvature components freely;
the existing scri `Project()` enforces null gauge values, not these tensor
constraints. Componentwise spatial differentiation, interpolation and KO do
not obey the nonlinear chain/product identities needed to preserve them.

For the interior continuum equations, with `delta=log(det(g))`,
`tau=g^ij A_ij`, `K=P+2Theta`, and `w=n_bar(Omega)`, contraction gives

```text
(dt-beta.grad) delta = -2 alpha tau
(dt-beta.grad) tau   = alpha (K-w) tau / Omega.
```

Zero algebraic constraints are invariant at continuum level; existing Z4
damping acts on other constraints. The ordinary production pure-gravity Z4c
path projects at its final explicit RK stage. Its dynamical-GR path additionally
projects intermediate stages. The masked Cartesian hyperboloidal adapter already
contains the same final-stage algebraic repair with rejection instead of floors.

## Existing snapshot evidence

Data: `output/hyperbolic-core-shell/puncture-duration-v3/n16/`, final `t=0.1`,
200 steps, `dt=0.0005`, `ko=0.5`. Exterior means `r>0.06461244037078456`, the
matched 1D horizon estimate used for this audit. RMS below uses coordinate cell
volume, not physical volume. Initial determinant error is zero; initial trace
error is at most `4.0e-15`.

| Region | Maximum determinant error | Maximum trace error |
| --- | ---: | ---: |
| Exterior | 2.9658411352e-4 | 1.0732104336e-3 |
| Cartesian core exterior | 5.6189009001e-10 | 8.5136525288e-8 |
| Exterior, r<0.56 | 6.1284565200e-7 | 5.6979846644e-5 |
| Gauge window, 0.56<=r<0.8 | 2.9658411352e-4 | 1.0732104336e-3 |

Both maxima occur at `r=0.703125`, in the middle shell, 2.5 radial cell widths
inside its outer seam. The determinant maximum is near a face axis; the trace
maximum is near an angular seam. Window RMS errors are `1.3604928991e-4` and
`6.1136109193e-4`. This is not solely a puncture defect.

Existing `puncture-v3/n12` and `n12-half-dt` snapshots both end at `t=0.02`.
Halving `dt=0.0005` to `0.00025` changes exterior determinant/trace errors by at
most `1.9692e-11`/`1.5974e-9`. This supports spatial rather than RK truncation as
the dominant source, but does not separate KO, transport and interpolation.

## Isolated helper and integration obligations

[algebraic_projection.hpp](../../src/z4c/hyperboloidal/algebraic_projection.hpp)
provides `KOKKOS_INLINE_FUNCTION` point helpers:

```cpp
auto result = ProjectAlgebraicTensors(g, A);
auto result = ProjectAlgebraicNativeState(values, compensation /* optional */);
```

They implement the existing ordinary repair, using the original metric in the
curvature correction:

```text
s = det(g)^(-1/3)
g' = s g
A' = A - g tau/3.
```

The result supplies status, validity, scale, and before/after determinant and
trace. Invalid/asymmetric/nonfinite tensors, nonpositive metrics, and numerical
overflow are rejected without partially updating either tensor. The native
wrapper also rejects a projected diagonal lost when packed as `g-I`.
Only native indices 1..12 are changed; optional compensated-RK residuals are
cleared at those indices only after success. Chi, P, Theta, independent Lambda,
and gauge fields are untouched. The helper validates tensor data only, not the
entire evolved state. Point arrays and optional compensation must not alias.

Future integration must choose and record stage placement, process volume and
scri consistently, rebuild derivatives/ghosts, and rerun scri null values,
geometric rates and dependent D. A separate projection of interpolated ghosts
would be an additional discretization choice. Never reset Lambda to the
metric-built connection. Final-stage projection alone does not constrain every
intermediate RHS or interpolated sample to roundoff.

This ordinary repair changes physical geometry by a truncation-sized amount.
Preserving reconstructed spatial metric and extrinsic curvature instead requires
`g'=s*g`, `chi'=s*chi`, `A'=s*(A-g*tau/3)`, `P'=P+Omega*tau`. Preserving Z then
requires a derivative-dependent Lambda transformation; the gauge variables and
coefficients also require consideration. That is a distinct formulation choice,
not what this helper implements.

## Reporting off the algebraic constraint surface

The existing evolved-constraint and reduced coordinate-sphere diagnostics assume
`tau=0`. For the physical ADM tensors actually reconstructed by FastFlow:

```text
K_ADM   = K + Omega*tau
H_ADM   = H_code + (4/3)*Omega*K*tau + Omega^2*tau^2
M_ADM_i = M_code_i + Omega*((tau/2)*d_i(log chi) - d_i(tau))
expansion_ADM = expansion_reduced - Omega*tau.
```

Actual FastFlow already contracts the full ADM tensors and needs no correction.
For the old n16 snapshot, the H correction has exterior RMS `3.8070617149e-4`;
the original H RMS is `0.05462853505`, versus `0.05465357914` after this correction.
These identities were independently checked with random tensor jets. They do
not reinterpret the old formulation norms as newly evolved ADM constraints.

## Bounded local validation

[test_algebraic_projection.cpp](../../tst/hyperbolic_shift_3d/test_algebraic_projection.cpp)
tests nonspherical SPD tensors, the exact ordinary repair, idempotence, rotation
covariance, exact CMC Minkowski identity, host/device-kernel parity, transactional
rejection, unchanged independent fields, and selected compensation clearing.
It then projects perturbed volume/scri data on a six-cell-per-block composite
grid and reapplies the existing closure, independently checking the time
derivative of the null relation. No evolution is performed.

Built directly using the existing `build-hyperbolic-mpi` scri-device target's
compiler flags and Kokkos libraries, without editing CMake. Local Serial results:

```text
rotation_covariance_max        6.6613381478e-16
scri_null_before_reclosure     1.2736914815e-3
scri_null_after_reclosure      1.1091860267e-15
interpolated_ghost_null        1.1102014958e-15
dependent_D_compatibility      2.1337098755e-16
null_time_derivative           7.7715611724e-16
```

This helper/test has not been compiled or executed with SYCL in this audit. The
separate unchanged Aurora preflight results do not validate its GPU behavior.

## Controlled MPI candidate

`hyperbolic_shift_3d_mpi_projected` compiles the same MPI runner with
`HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1`. The default MPI target remains the
unprojected control. After each compensated RK4 acceptance, the candidate
projects the owned volume and actual scri point tensors, clears compensation
only for metric/curvature components, gathers the accepted values, and reruns
the existing scri null projection and dependent-D calculation. Derivatives and
ghosts are rebuilt from that accepted state. Intermediate RK stages and
interpolated ghost tensors are not independently projected. A nonpositive or
invalid projection is a collective failure with rollback, never a floor.

The candidate writes checkpoint magic `HYP3D_MPI_V2`; the original uses `V1`.
Both executables reject a restart belonging to the other scheme. Reports name
the selected scheme, and histories now include determinant/trace residuals.
The exporter and restart adapter understand both formats without treating them
as interchangeable.

[Local evidence](reproduction/mpi-projected-local-v1/assessment.json) covers
one Minkowski step and four pulse steps at n8 on the Serial backend. The pulse
direct and restarted checkpoints are bitwise identical across all four arrays;
one- and two-rank runs also match bitwise. A cross-scheme two-rank restart rejects
collectively. Final pulse determinant error is `6.66e-16` and trace error is
`2.17e-18`. These checks establish implementation consistency only. The candidate
still requires SYCL preflight and a fresh stability trajectory before comparison
with the original control.
