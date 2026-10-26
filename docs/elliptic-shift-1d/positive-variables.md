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

## Conditional puncture obstruction in the symmetric Minkowski source

For the main Minkowski-reference lapse, assume a regular stationary trumpet
with alpha~a r^gamma (gamma>0), beta=O(r), finite Khat and Omega(0)=A(0)=1/2.
The transport and geometric-reference lapse terms then vanish at the inner end,
as does F(Khat+3), since F=O(alpha). The symmetric restoring source instead gives

    alpha_t(0) = -xi[alpha^2-A^2]/Omega at r=0 = xi/2.

Thus xi>0 with that *untapered symmetric Minkowski restoring term* cannot admit
the desired stationary collapsed-lapse endpoint under these regularity
assumptions. Diverging Khat would violate the assumed regular trumpet. This
conditional statement does not establish an obstruction to free Z4c or the
elliptic shift, and does not explain every numerical failure. It identifies a
gauge source that must change or taper for a regular main-reference trumpet.
The multiplicative form removes the constant term, but has the positive local
restoring derivative described above. The analytic CMC source control cancels
the constant on its own reference and therefore does not test this obstruction
for the main Minkowski gauge.

## Compiled AF coefficient evolution (API26)

The compiled engine now reconstructs the same positive AF chart and reference
jets as PunctureChart. The r=0 value evolves the asymptotic coefficient ODEs;
it is excluded from physical metric positivity checks and receives no regular
center parity, inner outflow guard or KO filter. Every r>0 curvature, Khat,
Theta and Lambda rate remains independent. Total Khat is reconstructed without
subtracting3, and the inner canceled R/S algebra matches the separately
audited Python formulas. Transport and puncture dissipation act on stored
coefficients after physical-rate conversion. Every RK stage solves the same
metric-derived shift quadrature. This path currently supports vacuum with
prefix quadrature and multiplicative restoring. Unsupported choices are rejected.

Both implementations now differentiate lapse deviation directly from the
reference deviation jets and expm1(V); subtracting A derivatives from total
alpha loses small outer-collar deviations. This is an algebraic arithmetic
repair. It changes no continuum equation or inner/scri condition.

All40 test groups pass (`evidence/ctest-v34.log`). The96 independent compiled/
Python AF cases cover two grids, both endpoint limits and transport choices,
three slicing families, two damping products and both source profiles.
Maximum RHS/short-RK differences are2.04e-12/8.89e-16; stage constraint and
Bondi monitors agree. An actual t6 wormhole trajectory has maximum physical
field difference3.10e-12 and is about20 times faster compiled at N129.
AF restart is bitwise reproducible in state, physical totals, grid/time/steps
and flux. These checks establish implementation agreement through a failed
evolution; they are not nonlinear stability or wormhole-to-trumpet admission.
The new frozen N513/1025/2049/4097 and independent timestep controls retain
the same physical-log gauge, independent fields and acceptance thresholds.

## Biased transport in the positive chart (API29)

Generic log chi and log(alpha/A) now support upwind transport in Python and
the compiled engine. Only the stored-variable advective derivative changes.
Its physical correction is beta*chi*(D_up-D_center)U or
beta*alpha*(D_up-D_center)V, with reference-profile derivatives evaluated
analytically. Other fields retain their ordinary independent biased transport.
Both regular-center layouts and coupled scalar rates/flux are checked. The
native physical equations remain the independent centered reference; an
additional exact centered/upwind identity agrees within4.1e-18.
All44 groups pass (`evidence/ctest-v37.log`), including96 positive-field
compiled/Python rate/RK/monitor cases. A compensated scalar/noise upwind
discrete-shift restart preserves state, low part and flux bitwise.

The predeclared positive-upwind-protocol-v1.json governs separate finite pulse
and coupled-collapse matrices. The scalar study distinguishes reduced positive
log strength and transport at a middle grid. Neither these tests nor the new
method's availability establish successful collapse.
