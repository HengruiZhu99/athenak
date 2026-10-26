# Active scientific work record

This goal remains active and incomplete. No production merge is authorized.

## Update: 2026-10-26, coupled scalar and central instability

Five CTest groups pass (`evidence/ctest-v5.log`), now including the canceled
nonlinear g=1 kernel, exact off-constraint angular Ricci/scalar geometry, and
analytic regular-center limits. The latest scalar+supported-noise checkpoint
test reproduces state, grid, time, step and integrated flux bitwise across an
aligned restart (`output/elliptic-shift-1d/coupled-restart-noise-v1`).

The revised lapse adds `-xi(alpha+A)(alpha-A)/Omega`, xi=1, as an explicitly
labeled gauge change. It substantially improves scri behavior; xi=0 remains a
failed control. Native primitive evaluation suffers central cancellation errors.
A separately implemented exact g=1 reduction cancels those analytically and
agrees with 24 nonlinear off-constraint native cases to7.6e-14. It preserves
independent curvature, Khat, Theta and Lambda evolution. No numerical
Schwarzschild/background RHS is subtracted.

With xi=1, sigma=4, tau=8 and origin KO coefficient .1, the published-shaped
amplitude .001 lapse pulse gives H RMS3.75e-9,6.37e-11,1.11e-12 at t=1 for
N256/512/1024; bulk orders5.88/5.84. Endpoint momentum order falls to2.31 on
the fine pair, failing the predeclared endpoint-order gate. At t=3 the fine
grid's central mode is growing (H RMS7.87e-10) although scri poles stay small.
These are not robust accepted evolutions. See the source-bound summary and
plot in `evidence/regular-center-review-v1/`.

Two attempted center repairs are rejected: direct chain-rule origin variables
fail earlier on refinement; an actual-origin grid with correctly tested analytic
limits fails positivity at t=2.256494 for N1025. The separate fourth-order
stencil control fails positivity at t=4.417098 after large central growth.
With stronger localized KO .5, N256/512/1024 reach t=6 with final H RMS
at or below8e-16 and M RMS below8e-15. The N512 amplitude .1 pulse also reaches
t=6 (H/M RMS1.4e-14/1.2e-14). These are improved short-duration controls;
convergence, elliptic/time/dissipation isolation and long runs still remain.
Supported smooth free
constraint errors and seeded nodal noise are now available via `--perturbation`.

Fully coupled scalar evolution is implemented, including the free-Z4c angular
Ricci contribution to the wave potential, all Einstein matter sources and a
nonlinear Hamiltonian initial-data solve. Fixed analytic wave RHS convergence
is sixth order. Amplitude .058 initial data give independent sixth-order ADM
constraint/energy checks, but strict Newton1e-11 reaches a double-precision
floor near1.6e-11 at N512. Candidate tolerance1e-10 is explicit; independent
tolerance/precision isolation remains outstanding.

The amplitude .01, N256 coupled pilot reaches t=3 with H/M RMS about
3.15e-12/3.84e-12. Initial mass .003132366825394506 minus RK4 stage-integrated
coordinate-time scalar flux .003132355606598225 is1.1219e-8. This is a useful
weak-dispersal pilot only. Bondi-time normalization, three resolutions,
timestep refinement, subcritical paper-amplitude runs, collapse, BH scalar
perturbations and tails remain unaccepted or unrun.

For a matching KO .5 weak-scalar sequence, N256/512 mass-minus-flux residuals
are2.13e-8 and5.35e-10 at t=3; N1024 is still running. Earlier KO .1 data
must not be mixed into this dissipation-controlled sequence. At amplitude .058,
N256 reaches t=6 but transient H/M RMS peaks .35/.47, scri pole5.35e-6, and
mass-minus-flux -1.40e-4: this fails acceptance despite its small final
constraints. N512/1024 refinements are in progress. The third-jet scri mass
diagnostic reports lower numerator jets explicitly; it is not admitted as a
Bondi mass extraction while these compatibility/roundoff errors are unresolved.

The source-linked literature inventory now has91 rows with exact retrieved
versions, including separate follow-up gauge/violation/solved-scalar/BH cases.
The2026v4 revision of2506.15311 agrees with earlier analytical charged tails;
the superseded disagreement is not a current finding. Parameter/figure audit
and all published-formulation controls remain unfinished.

Latest trajectories automatically preserve exact runnable sources and matching
libraries at startup. Older source-bound continuations use verified frozen
closures, including manually reconstructed, hash-matched source dependencies.
All failed variants and checkpoint ancestry remain available.

## User revisions

On 2026-10-25 the user authorized local one-dimensional builds/tests after Aurora
SSH authentication failed. The user then explicitly authorized small equation
changes, variable changes and boundary-method changes to address regularity,
with continued iteration until the full goal passes robustly. These revisions
supersede the corresponding restrictions in the preserved original objective.
They do not authorize hiding failures or projecting away differential constraints.

## Earlier component record (superseded where the update above is newer)

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

## Outstanding required work

1. Resolve the fine-grid central instability; repeat resolved pulse, violation
   and noise studies and test long-time behavior. The formal spatial chart
   below remains research, not an adopted time-closed method.
2. Complete published gauge/formulation controls, exact-scri/staggered comparison,
   elliptic quadrature isolation and temporal/endpoint convergence.
3. Validate coupled scalar evolution and constraint-solved data with principal
   resolution/time/elliptic/dissipation studies, followed by all required
   dispersal, collapse, BH perturbation, ringing/tail and mass-balance cases.
4. Derive puncture inner-end conditions and a puncture driver, then execute
   published trumpet controls and genuine wormhole-to-trumpet evolution.
5. Complete the reproduction inventory, physical diagnostics/flux/Bondi-time
   extraction, principal three-resolution and independent timestep studies,
   paper-length durations, and final assessment. All unrun tests remain unfinished.
