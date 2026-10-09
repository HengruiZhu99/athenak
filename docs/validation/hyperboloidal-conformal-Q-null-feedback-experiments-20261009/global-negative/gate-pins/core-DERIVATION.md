# Conformal-Q lapse plus null-residue source feedback: local exploratory gate

This is a new gauge combination, not a new storage variable or geometric
formulation. Every tensor equation remains the production C0 ConformalRHS,
with physical storage P=K_phys-2Theta_phys and kappa1=kappa_input/alpha,
kappa2=0. No BH subtraction, lapse/Omega floor or imposed Theta falloff is used.
The examined radii have S=1 and a=.5,.75,1,2. There is no native run in this gate.

## Existing controls and the distinct change

The requested harmonic conformal-Q lapse with bounded algebraic F0 and the
original preferred spatial projection already exists as
physical_trace_lapse=false, preferred_source=true. Its zero-jet pole has a
positive normal cubic root, 2.57170948731154 at a=1,kappa_input=5. This is the
versioned LowerOrderAudit/test_physical_gauge control, not a newly discovered
stable alternative. The current matched sigma0 rows reproduce it for attribution.

The earlier scratch null-feedback candidate used the physical-P lapse plus a
rederived temporal-source pole and preferred projection. Its twelve negative
poles/eight semisimple zeros did not ensure native pulse stability. In
particular that earlier projection also repairs the initial Einstein witness
null corner below: the present candidate is not the first control to do so.
Its actual original native receipts, helper and pole derivation are pinned by
prior-controls.json and are not rerun or changed here.

The distinct combination retains the original Q lapse/projection in the
harmonic collar and adds only the value-only null-residue beta pole. Unlike
the earlier physical-P projected candidate, the Q lapse retains one additional
compatible zero direction. That difference is a hypothesis to test, not an
argument for global stability.

## Lapse and the recommended inner blend

Let h=alpha_ref, B=beta.dOmega, Bh=beta_ref.dOmega, da=alpha-h,
D=alpha Bh/h-B and F=alpha^2+2(1-W)alpha. The original Q lapse is evaluated as

    Ralpha_Q = beta.[dalpha-(alpha/h)dalpha_ref]-alpha nu log(alpha/h)
    Salpha_Q = -F(P-Pref)+3[alpha+2(1-W)]D.

Thus no Q=(P-3wn)/Omega, no Q reference subtraction and no live-alpha quotient
in a Q numerator is evaluated. Near the reference D=Bh da/h-deltaBeta.dOmega
is used; for alpha/h<=.5 the direct D=alpha Bh/h-B retains its small
representable term. The logarithm uses the same robust branch as production.

The physical-P lapse parts are the unchanged algebraic production equations

    Ralpha_P = beta.dalpha-beta_ref.dalpha_ref-alpha nu log(alpha/h)
    Salpha_P = -F(P-Pref)-W xi(alpha+h)da
               -W[alpha deltaBeta+beta_ref da].dOmega.

Two explicitly named variants are checked. physical_inner=false is the global
original-Q lapse. The recommended physical_inner=true is

    Ralpha = (1-W)Ralpha_P+W Ralpha_Q
    Salpha = (1-W)Salpha_P+W Salpha_Q.

It is exactly physical-P for W=0 and exactly harmonic Q for W=1. The exact
W=0 branch short circuits the unused Q gauge evaluation. The beta regular
source remains the original preferred algebraic extension through 0<W<1.
No old preferred Box identity is claimed there after the alpha blend: the
actual temporal GH source changes, and the associated spatial GH source
changes by the lapse shift. Exact source/Box statements below apply only to
W=1. The blend modifies only lower-order value terms, including the reference
advection factors; it does not change the coupled principal derivative terms.

The helper never overrides xi. Nonlinear reference/transition tests use both
xi=1.5 and xi=1/a. The proposed target a=.5 uses the inherited xi=2 explicitly.
Outer W=1 pole/witness tests are xi-independent. There are no transition
finite-frequency matrices in this local core gate; those are a separate task.

## Source feedback and robust assembly

Set v=SmoothCutoff(r,.85,.95), with onset>=gauge_r1, so v>0 only where W=1.
Use V^i=Omega_i/sum_j Omega_j^2, G=chi gtilde^{ij}Omega_iOmega_j and
Nraw=G-wn^2, wn=-B/alpha. Relative to the original preferred gauge,

    Delta Salpha = 0,
    Delta Sbeta^i = v sigma V^i alpha^2(Nraw-Nraw_ref), sigma=5.

The spatial norm difference is factored exactly as

    deltaG=[(chi-chihat)ginvhat+chi(ginv-ginvhat)]^{ij}Omega_iOmega_j,
    alpha^2 deltaN=alpha^2 deltaG+D(B+alpha Bh/h).

This directly weighted numerator avoids division by the live lapse, and is
exactly zero on the analytic reference. The original preferred projection is
also evaluated as alpha^2 Delta directly. In particular

    alpha^2 F0 = beta.dlog(h)+nu log(alpha/h)-alpha Kbar_ref,
    alpha^2 H4 = (alpha^2 chi ginv-beta beta):Omega_hessian,

so no 1/alpha^2 projection intermediate is required. Shift gradient factors
use alpha^2 dlog(alpha)=alpha dalpha-alpha^2 dlog(h).

The production AssembleGaugeInterior helper ignores pole.beta. qnf::Assemble
therefore calls it once and adds each Sbeta^i/Omega exactly once. Native and
diagnostic wrappers must retain that explicit addition.

## Actual four-dimensional source identity

At W=1, the temporal source from the actual geometric and gauge time
rates is

    Gamma4^0+2Z4^0 = F0
    F0=[beta.dlog(h)+nu log(alpha/h)]/alpha^2-Kbar_ref/alpha.

Indeed -Salpha/alpha^3-(P-3wn)/alpha=-Omega Kbar_ref/alpha, so the potentially
singular P term cancels without forming Q or Theta/Omega. F0 is bounded in
the smooth Omega->0 limit for positive nonzero lapse; this is not a uniform
bound as alpha->0.

For the spatial projection, H4=(chi ginv-beta beta/alpha^2):Omega_hessian,

    Gamma4^i+2Z4^i = F^i,
    Omega_i F^i = H4-Omega What-v sigma deltaN/Omega,
    Box(Omega) = Omega What+2Z4^i Omega_i+v sigma deltaN/Omega.

Z4^i=.5chi(Lambda-Gammatilde)^i-beta^i Theta_phys/(alpha Omega).
The independent test constructs all four-dimensional metric derivatives and
Christoffels from the actual tensor/gauge rates, on off-constraint SPD states,
including the nonflat geometric transition .85<r<.95. This is an off-constraint
source extension, not exact Box(Omega)=Omega What on arbitrary vacuum fields.
If deltaN=Omega^2 deltaN2 and the relevant Z4 contraction is regular, the added
source is O(Omega). Those conditions are not imposed or proven preserved.
Tiny-alpha gauge-source tests do not assert bounded GH F0, positive energy or
uniform puncture hyperbolicity.

## Leading value-only pole and its limitation

Normalize time by a^2 and let K=kappa_input*a^2, C=a^2 deltaNraw,
T=a delta(P-3wn), E=a deltaTheta. At the reference boundary

    C=deltaChi-h_nn+2a(deltaAlpha+deltaBeta_n),
    T=a deltaP-3a(deltaAlpha+deltaBeta_n).

The actual full20 pole induces the autonomous normal block

    [ -2sigma       -4/3       4/3    ]
    [ 3sigma-3        1        K-4    ]
    [   -3           -2       -1-2K   ].

Its characteristic polynomial is

    z^3+2(K+sigma)z^2+[4sigma(K+1)-9]z+(8K-6)sigma-12K.

For K>0,sigma>=0 it is Hurwitz exactly when
K>3/4 and sigma>6K/(4K-3). For sigma5 this becomes K>15/14; kappa5/10 and
all four examined a satisfy it. The exact Routh identities are in check_gate.py.
The other nonzero factors in physical lambda units are

    (lambda+2/a^2)^2,
    (lambda^2+kappa lambda+2kappa/a^2)^2,
    lambda^2+kappa lambda+2kappa/a^2+4/(3a^4).

The full characteristic polynomial also has lambda^9. The rationally
reconstructed actual matrix has rank M=rank M^2=11: its nine zero roots are
semisimple, and the remaining eleven are strictly negative for the tested
parameters. Exact polynomial/nullity assertions apply to that reconstructed
analytic reference pole matrix, with its floating reconstruction error reported.
They are not an exact general nonlinear Omega0 assembly or a uniform PDE proof.
The compatible zero directions are linked by C=T=E=0; they do not supply
arbitrary independent lapse/shift boundary values. Nonfree angular jets,
normal first jets and the full A/Lambda regularity constraints remain separate.

## Initial Einstein gauge corner and prior candidate comparison

In the pure CMC outer region use the exact linear Cartesian gauge perturbation

    deltaAlpha=Omega, deltaBeta=-Omega n,

with every geometric/P/Theta perturbation zero. Its physical H/M/Z/Theta
constraints vanish identically, its initial actual R0 residue is zero, and

    deltaNraw=-a Omega^3+O(Omega^4),
    deltaQ=1.5 a^2 Omega^2+O(Omega^3).

The current actual kernel gives

    alphaDot0=1/a^2, betaDot_n0=0, PDot0=3/a^2,
    chiDot0=-2/(3a), hnnDot0=4/(3a),
    NrawDot0=QnumeratorDot0=0.

The sigma term is O(Omega^2) in betaDot for this witness. This is only the
initial scalar null/Q corner, not a closed first-jet hierarchy or persistence
claim. Full boundary Taylor compatibility is deferred to a separate map gate.

For comparison, the archived physical-P plus full preferred projection plus
sigma has d=1+2xi*a and

    alphaDot0=-d/a^2, betaDot_n0=(d+1)/a^2, PDot0=3/a^2.

It also has NrawDot0=QnumeratorDot0=0. That prior actual helper is checked by a
new cheap gauge directional finite difference here, without rerunning its old
full20/native sweeps. Thus its negative native evidence is not bypassed by
presenting the witness cancellation as a new stabilization mechanism.

The finite-Q counterexample also remains. P-Pref=.01Omega with the consistent
normal derivative gives Omega QDot->.02/a^2 and ThetaDot->-.02/a^2. Its Lambda
leading residue is nonzero; it lacks full first-jet/Einstein compatibility.
This establishes a smooth spacetime compatibility failure, not necessarily
finite-Q amplitude blowup, native failure, or a constraint-preserving closure.

## Remaining gates

Finite-Omega lower-order/frozen-frequency growth, full coefficient-aware/global
boundary dynamics, nonlinear Einstein/null first-jet propagation, energy and
BH/trumpet compatibility are unaccepted. Positive finite-frequency roots must
be retained and cannot be inferred away from negative leading poles. No native
compilation/evolution or production change is performed by this local audit.
