# Positive fields and separately labeled trumpet controls

These changes retain independent Ar, Khat, Theta and Lambda evolution. They
neither solve nor reset a differential constraint. All previous failed runs
remain distinct. The experimental branch is still incomplete.

## Regular-center positive chart

Let A=(1+r^2)/2. The optional invertible fields are

    chi=exp(U), c=chi-1=expm1(U),
    alpha=A exp(V), u=alpha-A=A expm1(V).

Apply the chain rule to the full nonlinear physical RHS, U_t=chi_t/chi and
V_t=alpha_t/alpha. The curvature, trace and constraint variables are unchanged.
The radial derivatives are

    c_r=chi U_r, c_rr=chi(U_rr+U_r^2),
    u_r=r expm1(V)+A exp(V)V_r,
    u_rr=expm1(V)+exp(V)[2r V_r+A(V_rr+V_r^2)].

The last two expressions are evaluated as deviations directly. Computing the
full alpha derivative and subtracting r or1 loses perturbations smaller than
machine relative precision. A1e-25 polynomial perturbation independently checks
that cancellation. Exact positive totals are supplied to both geometry and
scalar matter/wave equations; recovering them from1+c orA+u would discard a
collapsed total smaller than reference relative precision. A separate exp(-40)
RHS audit exercises that case. Underflow, overflow or nonpositive totals halt a
run. No lapse/metric floor is imposed.

KO acts on the stored fields after the physical-to-chart rate conversion.
Thus changing this chart also changes its finite-grid dissipation. The continuum
Einstein/gauge equations are the same, but finite-grid trajectories need not be
identical to primitive-field trajectories. Dissipation controls remain required.
The compiled implementation (currently ABI22) and Python implementation agree
in48 positive-field/center/scalar/endpoint/KO combinations, including stage
constraints, flux and Bondi nonaffinity. These are equation/implementation
checks, not a nonlinear stability proof.

## Already-trumpet chart and lapse-source control

`puncture-asymptotics.md` derives the CMC trumpet geometry. A separate optional
chart uses chi=r^2 X, X=exp(U)/A^2 and alpha=alpha_CMC exp(V). Define
ell=(log X)_r and ell_r=(log X)_rr. Then

    chi_r=2rX+chi ell,
    chi_rr=2X+4rX ell+chi(ell_r+ell^2),
    M_r=Omega(Ar_r-3Ar ell/2)-2(Khat_r+2Theta_r)/3+2rAr.

The canceled scalar-curvature combination is
2X+2rX ell+2chi ell_r-chi ell^2/2. The metric equation is evaluated as

    R_chi=chi(beta ell-alpha Ar),
    S_chi=chi(2alpha K/3-2r beta).

Other inner tensor expressions are algebraically rearranged in `g1_rhs.py`,
then independently compared with native tensor rates and constraints. This
cancels known leading1/r terms; it does not set curvature or constraints from X.
The trumpet chart is used only below r=.5 and joins the equivalent ordinary
expressions above. It has not cured the tested main-gauge constraint growth.

Choosing the analytic trumpet lapse as a *variable profile* does not choose a
black-hole evolution reference. The main lapse reference remains Minkowski.
The separately labeled `--lapse-reference CMC-trumpet` adds an immutable analytic
source G(r) to the lapse numerator only. With the reference deviations u_ref,
v_ref=C r/R^3 and beta_ref, the stationary lapse equation requires

    G=-r v_ref-beta_ref(Omega u_ref,r+r u_ref)+restoring_ref.

All reference derivatives and beta_ref are analytic initial-geometry quantities.
No numerical derivative, evolved field or numerical Schwarzschild RHS enters G.
This changes the gauge source, leaves the Einstein equations untouched and has
G=O(Omega^3), so the first scri lapse limit is unchanged. Both restoring forms
have independent native stationary-lapse audits. These labeled controls cannot
replace the required Minkowski-reference wormhole evolution.

Fixed-radius excision is another separately labeled already-trumpet control.
It uses one-sided radial differentiation below the horizon and halts whenever
any physical or gauge characteristic becomes incoming. The incoming-gauge
Robin prototype instead supplies the separately derived principal gauge mode,
still requiring physical/constraint outflow. Neither substitutes for a puncture
or a genuine wormhole-to-trumpet evolution.

The multiplicative restoring numerator -2xi alpha(alpha-A) has derivative
2xi A-4xi alpha with respect to alpha. At collapsed lapse this is positive;
an immutable stationary forcing does not alter that derivative. The symmetric
form -xi(alpha^2-A^2) has derivative -2xi alpha. The symmetric CMC/excision
control tests this distinction without changing the Einstein equations or
relaxing the outflow guard. It is not yet an accepted explanation for every
observed black-hole failure.

## Additional slicing control

The `blended-log` coefficient is

    F=(1-4Omega^2)alpha^2+8nu Omega^3 alpha.

At the center F=nu alpha, giving physical Bona-Masso f=2nu/physical_alpha;
nu=1 is exact1+log there. At scri F=alpha^2. Relative to the original coefficient,
the lapse numerator gains4Omega^2 alpha(alpha+nu r^2)(Khat+3). This preserves
Minkowski equilibrium but changes its interior linear gauge equations. Native
AF tensor and asymptotic audits pass. Its first fully coupled collapse trials
still have large/nonconvergent constraints, despite reaching the requested time.
They remain failed scientific controls.

## Accuracy samples and boundary quadrature

For a nonlinear constraint Q(q), a midpoint RK4 trial state
q_n+dt F(q_n)/2 differs from the exact half-step state by O(dt^2). Therefore
its Q need not have the fourth-order accuracy of an accepted RK4 step.
Every stage still solves the elliptic shift and checks positivity, scri poles,
clock, nullness and nonaffinity. Accepted k1 inputs additionally accumulate
bulk/endpoint constraint maxima. Analysis includes the last accepted report,
which otherwise has no subsequent k1 evaluation. It never promotes old sparse
histories to full accepted-step coverage. The instrumentation protocol was
recorded before inspecting its principal results.

The elliptic shift's inward option sums the same interval quadrature from
scri toward each physical node, avoiding subtraction of large prefixes.
Polynomial, large-dynamic-range and manufactured controls pass. This removes
one arithmetic cancellation but has *not* cured the measured endpoint
quadrature-isolation failure. Quadrature degree changes also change a spatial
operator; their measured effect must remain below the predeclared gate before
that control can be admitted. The independently derived local analytic scri
limit is being tested against the raw differentiated-numerator limit.

`trajectory.py` verifies checkpoint ancestry and source/library binding before
combining segments. Matrix and continuation drivers freeze source and binaries.
Report-cadence-only changes are recorded and checked bitwise in state/time/flux.
`analyze_principal_controls.py` compares physical fields, uses actual unequal
h ratios and records failed timestep cases. `assess_black_hole.py` re-extracts
horizons from saved profiles with a tested local polynomial interpolant; it
changes no evolved data and binds every analyzed profile by hash. Sampled
exterior constraints are labeled as sampled, not full-stage certification.
