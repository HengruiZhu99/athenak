# Active scientific work record

This goal remains active and incomplete. No production merge is authorized.

## Current coefficient-chart and independent-geometry audit

Twenty-five CTest groups pass (`evidence/ctest-v20.log`). The compiled engine is
ABI17, with640 Python agreement cases. The AF puncture coefficient chart is
implemented, including analytic reference derivatives, positive logarithmic
metric/lapse variables, free curvature/Khat/Theta/Lambda coefficients and a
true asymptotic endpoint. No center reflection, differential constraint reset,
Omega clamp or Schwarzschild RHS subtraction is used. Chart and rational-chart
restarts are bitwise identical on aligned steps; verified continuation retains
the exact source/library closure.

The old finite-cutoff incoming-characteristic failure is addressed by this
chart, but **wormhole-to-trumpet evolution remains unaccepted**. At M=.1 the
new runs still develop inner constraint growth and sharp lapse gradients.
Reference-lapse damping and two rational-lapse scale matrices do not cure it.
At M=1/3, N129/257/513 all reach t=6, but final H/M errors fail convergence
and every scri-pole peak exceeds its gate. Source-bound plots and outcomes
are in `evidence/wormhole-reference-damping-v1`,
`evidence/wormhole-rational-lapse-v1`, and
`evidence/wormhole-Mthird-convergence-v1`. Stronger dissipation and SBP controls also fail H/M convergence. The finite-cutoff
incoming-gauge Robin treatment supplies only a constraint-free principal gauge
mode, and correctly halts when physical characteristics become incoming near
t=.216. A collapsed-inner lapse-source control passes independent equation
audits but still fails nonlinear evolutions. All variants and their failures
are documented in `puncture-asymptotics.md`. Survival is not admitted as a
trumpet result.

An additional audit compares nonlinear geometric rates with physical ADM
Ricci/Hessian formulas on the constraint-satisfying Schwarzschild transition,
including nonzero radial tracefree curvature. Both native and canceled Z4c
rates agree to4.1e-13 (`evidence/physical-ADM-nonlinear-v1.json`). This rules
out a simple on-constraint tensor-reduction error in the tested initial slice;
it does not establish nonlinear stability.

The Bondi conformal frame is now explicitly checked: g_B=chi*g_bar and
Omega_B=sqrt(chi)*Omega give unit-sphere cuts and n_B^t=1/alpha. Its generator
nonaffinity is checked against independent2D Christoffel calculations. Frozen
source replay of the strong dispersal trajectories gives sampled maximum
nonaffinity3.93e-7/7.26e-8/9.86e-10 at N512/1024/2048; the largest integrated
finest defect is3.02e-11. These are sampled diagnostics, not all-stage bounds. ABI17 adds an independently
checked all-stage nonaffinity monitor; the matching three-resolution scalar
matrix is complete and bitwise identical in state/time/flux to its earlier
controls. Full-stage maxima are8.75e-7/7.41e-8/9.66e-9; the finest
6*max|kappa|=5.80e-8 passes the predeclared1e-7 affinity bound. Coarser
runs fail. The finest peak is almost ten times its sampled peak, confirming
that sparse replay cannot certify stage maxima. Source bindings and
instrumentation-equivalence checks are in `evidence/scalar-Bondi-allstage-v1`.
Its conservative supplemental affinity gate was recorded before inspecting
the new all-stage trajectory results (`affinity-acceptance-v1.json`).
Coarse output quadrature of the sharply peaked scalar flux differs materially
from its RK-stage accumulator, so the latter remains the mass-budget quantity.
See `evidence/scalar-Bondi-time-n{512,1024,2048}-v1`.

The small .001 gauge pulse has completed t=10,000 at N256 and t=3000 at
N512. N512 is continuing to10,000 with its exact frozen source and binary;
N1024 has completed1000 and is continuing to3000. These older trajectories have sampled diagnostics.
Their small amplitude and incomplete finer durations do not constitute a
published nonlinear pulse reproduction.

## Previous puncture and gauge preparation

Twelve CTest groups pass (`evidence/ctest-v15.log`). Compiled/Python agreement
now includes320 cases with one-sided puncture stencils and the multiplicative
restoring gauge. An ABI check prevents mismatched wrapper/library layouts.

The nonlinear scri gauge audit independently matches native tensor rates to
7.4e-13. In the previous spatial chart, xi=1 requires C0=2L0, then
C1=C0+11C0^2/4, with a further C2 relation saved in
`evidence/nonlinear-gauge-corners-v1.json`. For a CMC Schwarzschild exterior,
chi-1=-M Omega^3/2 implies delta-alpha=-3M Omega^3/4 at this order.
A scalar initial-lapse control imposes initial chi_t=0 and matches the exterior
stationary lapse to convergent accuracy. This necessary preparation **does not
cure the transient scri gate failure**; its runs are retained separately.
Published damping controls and the quadratic restoring variant also fail the
finite-pulse scri gate, so none is relabeled as an accepted repair.

`wormhole_isotropic.py` constructs a genuine two-ended wormhole with g=1,
H=n(R)=s(R)R in the outer transition, a time-symmetric isotropic inner end,
physical pre-collapsed lapse psi^-2, and a stationary hyperboloidal exterior
lapse. The Minkowski evolution reference is unchanged. The512-point data
have H/M below2.3e-13/3.9e-14 and bulk invariant mass error1.24e-11.
The96/128 coordinate quadratures agree to9.4e-15. Analytic exterior geometric
rates vanish to1.1e-15 without numerical RHS subtraction.

The puncture driver uses one-sided stencils and difference-form derivatives,
never reflected center values. The regular-center KO filter is forbidden there.
Its first N1024 pilot reaches t=.1 with mass at r=.5 equal.1 to3e-14 and
horizon mass.0999965. However N512/1024/2048 outflow-only continuations halt
at t=.2042/.2132/.2156 when an inner gauge characteristic becomes incoming.
These are inner-boundary validity failures, not a demonstrated trumpet.
The stage monitor records the positive speed and preserves the last completed
state. Initial geometry and source-bound failures remain archived.

The next inner treatment is derived in `puncture-asymptotics.md`. For
chi=r^4X,alpha=r^2A,A_rr=r^2B,Khat=r^2k,Theta=r^2tau,Lambda=L,
beta=b1r+..., the native tensor equations independently verify the finite
coefficient ODEs there with second-order limiting convergence. All six
coefficients remain free. A true r=0 asymptotic coefficient endpoint can
therefore replace the finite-cutoff outflow assumption without regular-center
parity or invented incoming data. The subsequent implementation and failed evolution controls are recorded in
the current update above.

## Latest audit: compiled integration and complete-stage diagnostics

The compiled g=1 RK4 engine uses the supplied FD/quadrature weights and the
same equations as Python. Across64 combinations of origin grid, scalar/vacuum,
Theta source, dissipation, endpoint formula and optional gauge-corner penalty,
RHS agreement is better than8.6e-14 and20-step state agreement better than
1.2e-16. A full weak-scalar trajectory differs by2.46e-14 with flux difference
8.7e-19. Observed runtime falls from30.3s to1.91s in that pilot; this is a
specific comparison, not a universal performance claim. Every binary is frozen
with its run. Exact restarts include the compiled binary hash.

Eight CTest groups pass (`evidence/ctest-v9.log`). The tests include an
independent native-tensor check of the complete analytically differentiated
scri limit. R/S decompositions are nonunique up to Omega multiples; comparing
S derivatives alone is incorrect. The complete R-S' limit converges against
native source differentiation with orders7.60 and7.82.

The latest suite has nine passing groups (`evidence/ctest-v12.log`). Optional
one-point biased advection passes polynomial/symbol checks and extends the
compiled/Python comparison to128 cases. It does not repair the finite-pulse
scri failure (N512 peak pole1.54e-5; N1024 peak1.02e-6), and is not silently
adopted as a successful method. Centered transport remains the default.

The `.058` initial-data solve at N1024 stalls in double precision near1.13e-10.
A50-digit Decimal iterate/residual with double Newton corrections reaches
1.25e-21 at the unchanged requested tolerance1e-11; its mass agrees with the
double iterate to displayed precision. Rounding the accepted solution back to
double gives residual2.96e-10, explicitly reported separately. No tolerance
was loosened. Failed initializations now preserve their own source, iterate,
Newton history and failure receipt before evolution.

The small lapse pulse has reached t=100 at N256/512/1024, CFL.1, with final
H/M RMS below1e-43. These older runs have sampled diagnostics; they do not
certify all-stage peak gates. The later N256 continuation reaches10,000; matching finer-duration and
finite-amplitude published controls remain unfinished.
The matching short control gives all-field bulk/endpoint self-convergence
orders5.06/5.13, RK4 orders4.003/3.969, and standard timestep error about
2.9e-5 of the coarse spatial difference (`evidence/regular-controls-v1`).

Important negative results remain. The finite amplitude.1 pulse exceeds the
scri-pole gate during propagation; a.1 output cadence substantially misses
the peak. New compiled runs monitor positivity, H/M/Theta/Z RMS, nullness,
clock, pole and regularity maxima at **every RK stage**. A separate test
compares these monitors with Python stage diagnostics. Quadrature degree5
can trigger a large scri failure; degree7/9/11 differences must be assessed
against spatial error at the same resolution. The finest N1024 differences
are near roundoff in the bulk but still material relative to the small
endpoint spatial error. This isolation gate is not yet admitted.

Analytic pole-numerator derivatives (`--endpoint analytic-g1`) are verified but
do not cure finite-amplitude scri errors. Stronger value penalties are not a
robust repair either. A weak first-normal-jet gauge penalty has the correct
isolated directional damping Jacobian, but **all nonzero tested strengths
fail in the coupled system** by t=.36-.57. It is retained as a rejected
variant, never the default. This is another example where a component audit
does not establish coupled stability. Next investigate published upwind
advection and an energy-compatible/regular-variable boundary treatment,
rather than declaring a scalar damping calculation sufficient.

Refined `.058` dispersal with50-digit initial solves is in the all-stage
512/1024/2048 matrix. N512/1024 stage H peaks are.0111/.000351 and their pole
peaks1.49e-6/1.42e-7, so both fail the pole gate despite tiny final errors.
N2048 is finishing. The `.07` N512 collapse pilot develops horizon roots but
fails at t=1.66764 with H/M peaks23/103; horizon formation is **not accepted**
from that unresolved run. No valid puncture evolution is implemented yet.

The smooth supported-constraint sequence128/256/512 reaches t=10 and decays.
The original fixed-amplitude nodal-noise sequence reaches t=6 but its finest
Theta amplification129.5 fails the declared100 per-component gate. It is
retained as failed. H initially scales like k^2 delta-chi, while Theta_t
contains alpha H/(2Omega); over a wave time1/k it can generate O(k delta-chi)
Theta from initially unrelated noise. A distinct derivative-balanced dataset
uses chi amplitude proportional h^2 and independent curvature/Khat/Theta/Lambda
amplitudes proportional h. Its separately declared protocol is
`constraint-noise-gates-v2.json`, not a relabeling of the original test.
The four128/256/512/1024 trajectories finish with small all-stage poles;
per-component acceptance analysis is saved and passes all four resolutions
(`evidence/balanced-noise-v1`). The original unscaled failure remains unchanged.

The all-stage `.058` scalar matrix is complete: peak H RMS for512/1024/2048
is.01107,.0003515,1.152e-5; mass-minus-flux is-6.99e-6,-2.53e-7,-8.19e-9.
The latter converges at roughly fifth order. Stage pole maxima are1.49e-6,
1.42e-7,1.26e-8; only the finest passes the pole gate. This is encouraging
resolved dispersal evidence, not admission of the full three-grid case.
The plotted sampled constraints are explicitly labeled as sampled; acceptance
uses the separate every-stage maxima (`evidence/scalar-allstage-v1`).

The compiled checkpoint continuation helper was tested with exact frozen
Python sources and both matching libraries. An N256 small-pulse trajectory
is now continuing from t=100 to1000 using its verified original source/binary,
without changing its equations or timestep. Longer paper durations still remain.

That N256 continuation has now finished t=1000, with finite positive fields
and perturbations at subnormal roundoff. Its exact-source continuation to
10,000 is running, as is N512 t100-to1000. Both are sampled-diagnostic long
controls; all-stage short-run gates are assessed separately.

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
4. Repair the implemented puncture chart's nonlinear constraint growth, then
   execute published trumpet controls and genuine wormhole-to-trumpet evolution.
5. Complete the reproduction inventory, physical diagnostics/flux/Bondi-time
   extraction, principal three-resolution and independent timestep studies,
   paper-length durations, and final assessment. All unrun tests remain unfinished.
