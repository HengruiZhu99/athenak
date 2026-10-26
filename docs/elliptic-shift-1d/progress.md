# Active scientific work record

This goal remains active and incomplete. No production merge is authorized.

## User revisions

On 2026-10-25 the user authorized local one-dimensional builds/tests after Aurora
SSH authentication failed. The user then explicitly authorized small equation
changes, variable changes and boundary-method changes to address regularity,
with continued iteration until the full goal passes robustly. These revisions
supersede the corresponding restrictions in the preserved original objective.
They do not authorize hiding failures or projecting away differential constraints.

## Implemented and verified components

1. Isolated baseline snapshot and experimental branch; source and Kokkos revision
   preserved, large generated outputs excluded and inventoried. Portable linear
   specification copied with provenance; it remains a linear claim only.
2. Nonlinear regular-center elliptic shift derived from the metric connection.
   Five automated checks pass, including independently discretized BVP and native
   tensor comparisons. No replacement of evolved Lambda by a metric derivative.
3. Independent initial-time ADM checks for a published Gaussian lapse pulse:
   curvature/trace rates converge at approximately sixth order through N1024.
4. Hyperboloidal Schwarzschild wormhole **initial data**, not an already-trumpet
   replacement: isotropic time-symmetric interior, smooth exterior height function,
   physical pre-collapsed lapse psi^-2. Exact-jet checks on405 points give
   max H=4.82e-12, M=2.76e-14, Z=6.21e-14. The inner radius diverges as1/r.
5. Independent fixed-Minkowski scalar propagation and center reflection:
   second-order upwind characteristics/RK4, N200/400/800. Scri waveform max error
   at N800 is .003456; measured orders1.898 and1.984. This passes its declared1%
   preliminary control. It is not a coupled Einstein-scalar result.
6. Bitwise state/time restart for an aligned N64 checkpoint. Separate timestep
   refinement of the failed spatial method gives RK4 order3.9945.
7. Initial literature inventory:68 explicit rows from the six requested sources
   and mandatory extensions, plus five follow-ups. Full follow-up extraction and
   figure-axis parameter/duration extraction remain unfinished. No published row
   is marked reproduced.

## Preserved failures and interpretation

- Raw l'Hopital boundary updates fail even for Minkowski at N64 near t=.05984.
  The discrete dominant mode is localized at scri and grows as1/h.
- Regular-quotient extrapolation reduces but does not remove this failure.
- Weak pole/regularity penalties with sigma4,tau8 preserve the unperturbed N64
  background to about5e-13 through t=3, but pulse runs fail scientific gates.
- The coarse compact pulse N32/64/128 study is underresolved; it shows no spatial
  convergence and must not be used to infer a continuum obstruction. Its failures
  and plots remain. A separately implemented compatible-trace candidate also fails
  and is not adopted: at t=.7504 its pole maximum is7.43 despite tiny value-trace
  residuals. Value compatibility alone is insufficient for normal-jet compatibility.
- Resolved published-Gaussian N256/512/1024 runs at t=.1 show H RMS
  2.95e-8,4.70e-10,7.36e-12 and M RMS4.64e-8,7.41e-10,1.17e-11;
  scri poles remain near roundoff. Their exact source-bound checkpoints were
  continued to t=.5. Then scri poles are .002815,.000565,.001172 and M RMS
  .000560,.0000414,.000217: endpoint acceptance still fails nonmonotonically.
  Refinement resolves the initial pulse but does not repair this boundary method.
- Every trajectory, command, source/library binding, failure and checkpoint parent
  is retained under ignored output/elliptic-shift-1d. The continuation helper copies
  and verifies the exact parent source closure and preserves a matching library;
  it never relaxes source compatibility merely to resume a run.

## Current regular-variable derivation

A new nonlinear spatial chart for g=1 has been verified symbolically. With
x=Omega, chi=1+x C, K=p+2Theta, and A_rr=(1-2x)B,

    P0=C_x/2-C/2-C^2/8,
    E0=4C_x/3-5C^2/6,
    P1=(9C^3+18C^2-18C C_x-20C_x+4C_xx)/8,
    K=-3-2x C+x^2P0+x^3P1+x^4 P_free,
    B=2C/3+xE0+x^2 A_free.

For arbitrary regular free functions this gives H=x^4 H_regular and
M_r/r=x^3 M_regular. At scri, letting E1=A_free(0) and
A_free_x(0)=a_free, the Jacobian from (P_free(0),a_free) to(H4,M3)
is[[-4,0],[8/3,-1]], determinant4. Thus H/M errors remain independent;
Theta=x^4 tau and Lambda=x^3 ell retain the other constraint errors.
This is an invertible differential chart in the interior, not a constraint solve.

**Not yet a time-closed chart:** a long-double Taylor substitution into the actual
nonlinear kernel exposes further gauge corner conditions. Arbitrary C(0) and
lapse normal jets fail lower-order compatibility. For the supplied test data,
chi_t(scri)=C(0)-2L(0) when alpha=A+xL; the reference lapse's raw clock rate is
also constrained. The formal constraint-corner matrix has full rank, and the
zero-gauge case recovers the linear-note amplitudes, but the finite-gauge tests
are not admissible yet. Do not describe these algebra tests as a completed
nonlinear closure. Derive and verify the gauge corner hierarchy and transformed
RHS before evolving the new chart. A collar-only variable chart can avoid
ill-conditioned Omega differentiation at the regular center without slaving any
bulk curvature/constraint function or the shift.

## Next required work

1. Complete the new chart's nonlinear gauge and subsidiary time-corner audit,
   implement analytically factored RHS/endpoint limits and appropriate origin
   discretization, then repeat resolved pulse, violation and noise studies.
2. Complete published gauge/formulation controls, exact-scri/staggered comparison,
   elliptic quadrature isolation and temporal/endpoint convergence.
3. Implement and validate coupled scalar evolution and constraint-solved data.
   The independent oracle is only the first prerequisite. A radius-weighted field
   psi=R phi is promising: exact spherical geometry gives
   box_2 psi-[2m/R^3-Ricci_theta_theta/R^2]psi=0. Dropping the Ricci term is an
   off-constraint modification and must be audited/disclosed if used.
4. Derive puncture inner-end conditions and a puncture driver, then execute
   published trumpet controls and genuine wormhole-to-trumpet evolution.
5. Complete the reproduction inventory, physical diagnostics/flux/Bondi-time
   extraction, principal three-resolution and independent timestep studies,
   paper-length durations, and final assessment. All unrun tests remain unfinished.
