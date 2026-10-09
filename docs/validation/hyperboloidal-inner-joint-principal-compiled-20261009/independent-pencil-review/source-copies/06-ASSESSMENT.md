# Inner moving-puncture feasibility, with the Minkowski reference retained

Source-only pencil assessment, 2026-10-09. No compiler, symbolic/numerical mathematics, kernel query, black-hole evolution, production change, or black-hole RHS subtraction is made. This supplements the preserved stationary calibration; it does not select a gauge. The user's later target includes a wormhole-to-trumpet transition while retaining the Minkowski hyperboloidal reference throughout.

## Actual inner equations and assumptions

The exact *geometric* core has Omega=1, reference alpha=chi=1, reference beta=P=Lambda=0 and all reference spatial derivatives zero. The gauge core additionally has W=0. On the Einstein sector Theta=Z=0, and with lapse_inner=0, both existing lapse flags reduce to

    D0 alpha = -alpha(alpha+2) K,
    D0 beta^i = mu0 alpha^2 chi Lambda^i - eta_inner beta^i,
    mu0=3q0/4=3/8,
    D0=partial_t-beta^j partial_j.

This is modified Bona-Masso slicing f=1+2/alpha, not exact standard 1+log. The connection is the actual contracted conformal connection on the Einstein sector; an independently assigned Lambda would introduce spatial Z. The production defaults have eta_inner=0. These equations apply only on the exact plateau, not merely wherever the gauge weight vanishes in a nonflat geometric layer.

Assume a radial, differentiated power/polyhomogeneous class at r>0 tending to the puncture: alpha=a(t)r^p(1+o(1)), chi=c(t)r^m(1+o(1)), beta^i=b(t)x^i+o(r), K=K0(t)+o(1), and bounded nondegenerate conformal metric with Lambda=O(1/r). Derivative bounds are part of this assumption; bounded metric values do not supply them. For a trumpet take p>0,m=2,c>0. The limiting puncture is not an ordinary positive-lapse/positive-chi interior point.

## Stationary and dynamical leading balances

For m=2,

    mu0 alpha^2 chi Lambda = O(r^(2p+1)) = o(r),
    beta^j partial_j beta^i = b^2 x^i+o(r).

The leading shift equation is therefore bdot=b(b-eta_inner). Stationarity with b>0 requires eta_inner=b. With the default eta_inner=0, this regular stationary class has no positive-b endpoint. Matching positive damping removes only this necessary leading obstruction.

The lapse gives

    pdot=0,   adot/a = b p - 2 K0,

provided the leading coefficients and finite K0 remain in the stated differentiated class: a nonzero pdot would leave an unmatched log(r) term. The actual default lapse_inner=0 is essential here. If a constant nu_inner>0 is enabled, the exact core also has -nu_inner alpha log(alpha), since alpha_hat=1. Matching log(r) instead gives pdot=-nu_inner p and adot/a=b p-2K0-nu_inner log(a). Thus a stationary collapsed leading power p>0 requires nu_inner=0 in this class, or a separately justified source. Restoring toward a hidden BH reference is not an allowed remedy.

The core conformal-factor equation is

    D0 chi = (2/3) chi [alpha K - partial_i beta^i].

Consequently mdot=0 and cdot/c=(m-2)b. In particular the leading trumpet coefficient c is stationary in this class. For a wormhole m=4 it can change in amplitude, but its strict r->0 exponent cannot switch smoothly to two at a finite time under this ansatz. This does **not** exclude a wormhole-to-trumpet transition on finite-radius grid cells. It distinguishes the finite-time puncture limit from the finite-resolution late-time exterior limit, and flags where an asymptotic expansion may cease to be uniform.

For the exact stationary Schwarzschild BM foliation already derived, R0/M=1.3195497562..., p=1.0607696620..., M b=0.5442012448..., and p b=2K0. For M=.5 the leading damping calibration is eta_inner=1.0884024895... . These are values from the preserved independent calibration, not a new calculation. Even for that foliation, an isotropic metric has Lambda=0 and full stationary shift balance requires eta(r)=partial_r beta^r. Its profile is not constant. A constant eta=b_end leaves a subleading residual O(r^(1+p)); a different radial coordinate/conformal metric might admit a stationary constant-eta driver, but has not been constructed.

## What the weighting does during a transition

The instantaneous connection response is proportional to alpha^2 chi. At trumpet scales it is parametrically weaker than the O(r) advection/damping response. For wormhole chi~r^4 and alpha~r^s with s>=0, the same Lambda bound gives a driver O(r^(2s+3)). A collapsed lapse therefore further suppresses this coordinate response at fixed small radius. With zero initial b the leading reduced ODE remains b=0; a nonzero shift can nevertheless be generated at finite r, by subleading terms, or outside this asymptotic class. It would be incorrect to turn this scaling into a theorem that formation cannot occur, or that the whole shift freezes.

In an isotropic trumpet, relative to advection, physical/light and current shift characteristic speeds scale as r^(p+1), while the lapse speed scales as r^(1+p/2). Advection -b r dominates these at sufficiently small r. The current driver does not retain the O(1) coordinate shift speeds of the usual conformal Gamma-driver. This changes which gauge information can leave the puncture neighborhood and demands a separate inner discretization/characteristic audit; it is not itself an instability proof.

The relevant physical mechanism is supported by Brown's analysis: ordinary moving-puncture Gamma-driver gauge modes include propagation in the regular conformal geometry, while finite-resolution points move away from the second wormhole end. That BSSN/standard-1+log result motivates the comparison but does not establish it for this Z4c/modified-BM/layered gauge. See [Brown, arXiv:0908.3814v2, Eqs.10g-h and Secs.I/III](https://arxiv.org/html/0908.3814) and [Brown, arXiv:0705.1359](https://arxiv.org/abs/0705.1359).

## A constant response is a principal change, not a damping fix

A direct conformal connection response G0(Lambda-Lambda_hat), G0>0, in place of mu0 alpha^2 chi(Lambda-Lambda_hat), is physically motivated by connection freezing/conformal Gamma-driver response. It must be written directly with bounded coefficients; evaluating mu=G0/(alpha^2 chi) creates unnecessary singular arithmetic. It is not algebraically the standard two-equation Gamma-driver with an auxiliary B field.

A constant response also changes regularity requirements. Generic Lambda=O(1/r) now produces an O(1/r) force, incompatible with a regular beta=O(r) stationary expansion. Such a class needs cancellation of those connection residues; if Lambda=lambda1 x+o(r), stationarity requires G0 lambda1=eta b-b^2. This relation must come from the evolved conformal metric, not an independent connection assignment. Thus a stronger driver offers a route to O(r) response without tuning eta=b, but imposes another coordinate/metric condition and supplies no all-order solution by itself.

The actual constrained20 core symbol shows why replacing only this coefficient is inadmissible without new gates. In the existing normalization let mu_eff=G0/(alpha^2 chi). With epsilon_alpha=epsilon_chi=0 its scalar shift speed squared is q=4mu_eff/3 and the lapse speed squared is f=1+2/alpha. The preserved actual-kernel proof exhibits defective scalar collisions at q=1 (f=3,mu_eff=3/4) and q=f=3 (mu_eff=9/4); the vector mu_eff=1 collision alone is semisimple. For G0=3/8 and a continuous collapse from alpha=chi=1 toward zero, q goes from 1/2 to infinity and encounters q=1; chi(alpha^2+2alpha) also goes from 3 to zero and encounters the lapse collision value 1/2. The collision condition is not cured by changing algebraic eta.

A future principal-adjusted option would therefore have to derive the shift-gradient couplings and, if needed, the slicing interpolation jointly. A bounded direct coefficient C_beta(r,alpha,chi) can interpolate a constant core response to the harmonic outer alpha^2 chi response, while the reference-deviation terms keep the same Minkowski fixed point. But that interpolation alone is no completeness proof. The existing harmonic endpoint requires the coupled coefficients (f,mu,epsilon_alpha,epsilon_chi)=(1,1,1,1/2); its complete light-cone basis must be retained. The whole positive-alpha/chi transition, oblique/SPD symbols, repeated-root eigenspaces, characteristic bounds and nonlinear reference identity would need fresh actual full20 checks. A two-equation standard Gamma-driver changes the state and mode count and requires a different gate. Neither choice is selected here.

## Consequences for later acceptance

The current weighted core is not justified as a general moving-puncture transition driver merely by having a singularity-avoiding lapse. Its default eta=0 has the stated necessary stationary obstruction; calibrated eta is only a local prerequisite. A constant or principal-adjusted core response is a reasonable *research comparison*, but the known scalar defects prevent immediate adoption. No uniform diagonalizer or hyperbolicity through alpha=chi=0 is claimed.

Later BH tests must retain the mass-zero Minkowski reference and physical-P evolution, derive independent wormhole data, and resolve the core and transition. They must measure areal-radius plateau, proper-distance growth, chi/r^2, lapse exponent, beta/r, actual full RHS decay, constraints and horizon/mass invariants. Existing span2.1 core-count examples have no exact-core cells at N24/36 and only eight at N48; the later span2.2 wave-map grids similarly do not resolve the .05 core through N32. Current Minkowski gates remain the prerequisite. No stationary outer-mass audit, BH source subtraction, BH inner blend or new native candidate is attempted.
