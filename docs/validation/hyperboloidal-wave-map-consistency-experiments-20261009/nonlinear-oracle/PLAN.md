This fresh oracle will test a localized finite-amplitude, exactly flat physical metric against the physical reference wave-map geometry. It will not propagate data, query an actual Z4c/source kernel, infer a scri limit, or establish nonlinear or black-hole stability. The same fixed Minkowski hyperboloidal reference is retained. The later single-hole requirement remains a wormhole-to-trumpet interior transition with that Minkowski reference throughout.

Execution is held until the parent reviews this fixed plan. Source preparation is permitted. The root derivation and the newly frozen higher-reference oracle are pinned in plan.json. Any failed attempt will retain its recipe, exact source, command, inputs, outputs and status in a fresh directory. No prior frozen bytes may change.

Use S=1, a=.5, geometric r0=.05/r1=.95 and the captured .9 width literal, interpreted as exact binary64 reference coefficients. The physical coordinates and Gaussian parameters have units of S. Set u0=-.5 S, sigma=.35 S and epsilon/S=0,.025,.05,.1. Let a^A=(0,0,0,1), and define

    Y^A(X)=X^A+epsilon*a^A*phi(X),
    phi(T,X)=sigma*[F(T-R)-F(T+R)]/R,
    F(s)=exp(-((s-u0)/sigma)^2), R^2=X^2+Y^2+Z^2.

phi is dimensionless, so epsilon has length. The center value is -2sigma F'(T); the perturbation is smooth there. A division-free equivalent function is

    q=(T-u0)/sigma, rho=R^2,
    phi=4q*exp(-q^2-rho/sigma^2)*0F1(;3/2;q^2*rho/sigma^2).

The hypergeometric function is sinh(z)/z for z=2qR/sigma, including its regular value at zero. Direct T/rho derivatives through total order three and the exact radial chain rule supply the Cartesian phi jets. The wave equation is checked independently as -phi_TT+6phi_rho+4rho*phi_rhorho=0, and against the original outgoing-minus-ingoing formula away from the origin. The regular-center value and odd spatial jets are checked explicitly. No finite differences or truncated Taylor padding are used.

The reference embedding is Yhat^0=t+h(r), Yhat^I=x^I/Omega(r). Use h=0 in the exact Cauchy core and

    h_r=b*L/(alpha_hat*Omega^2), L=Omega-r*Omega_r.

Integrate h_r once using mpmath tanh-sinh on the fixed panels listed in plan.json. Derivatives h through three come from the analytic integrand and its first two derivatives, not numerical differentiation of quadrature. In the exact outer branch use h=sqrt((r/Omega)^2+a^2)+C, with C fixed at r1 by the transition integral. Evaluate h-R there as C+a^2/(sqrt(R^2+a^2)+R). Compare both precision runs and the analytic outer derivative/matching identities. No adaptive choice of panels or tolerance after results is permitted; failures remain failed.

At each prescribed finite spacetime point, T=Yhat^0, X=Yhat^x, Y=Yhat^y, and solve the scalar equation

    z+epsilon*phi(T,X,Y,z)=Yhat^z.

The mean-value bound |phi|<=2sqrt(2/e)<2 brackets every root in [Yhat^z-2|epsilon|,Yhat^z+2|epsilon|]. A safeguarded scalar solve starts from the epsilon=0 branch. Check the scalar residual, precision agreement and local orientation D=1+epsilon*phi_z>0. Record D and inverse-map conditioning; no global injectivity claim follows from sample invertibility, and a nonpositive D or failure to follow the same branch fails the prescribed case without reducing epsilon. Epsilon=0 is handled exactly.

Writing J^A_B=delta^A_B+epsilon*a^A*phi_B, its rank-one inverse is I-epsilon*a^A*phi_B/D. The implicit jets are

    X_a=Jinv*Yhat_a,
    X_ab=Jinv*[Yhat_ab-epsilon*a*phi_BC X_a^B X_b^C],
    X_abc=Jinv*[Yhat_abc-epsilon*a*(phi_BCD X_a^B X_b^C X_c^D
                  +phi_BC*(X_ab^B X_c^C+X_ac^B X_b^C+X_bc^B X_a^C))].

These are full spacetime jets, including mixed time/spatial derivatives and their permutations. The metric is g_ab=eta_AB X_a^A X_b^B. Product rules then give every g_ab,c and g_ab,cd directly. Only map order three is required for physical metric order two. Check the differentiated implicit identities and their symmetric permutations. The linear active-coordinate generator is xi^A=-a^A*phi, with the minus sign from the inverse map.

Construct gbar=Omega^2*g. Extract physical lapse ell=sqrt(-1/g^00), beta^i=gamma_phys^{ij}g_0j, and gamma_phys_ij=g_ij. Use the repository sign

    Kphys_ij=-(partial_t gamma_phys_ij-Lie_beta gamma_phys_ij)/(2ell).

Extract Penrose alpha=Omega*ell, bargamma=Omega^2*gamma_phys, chi=det(bargamma)^(-1/3), gtilde=chi*bargamma, P=Kphys and Theta_phys=0. Stored Atilde=Omega*chi*(Kphys_ij-gamma_phys_ij*Kphys/3), and Lambda^i=GammaTilde^i since Z_i=0. Derivatives of all coefficient factors are retained. Verify the connection by both the contracted Christoffel definition and -partial_j gtildeInv^{ij}. Check ADM reconstruction, determinant one, A trace zero, physical H/M zero, and the Penrose/physical lapse distinction.

Export complete consumed spatial jets: Omega through two; alpha, beta, chi and gtilde through two; Atilde, P and Lambda through one; Theta_phys identically zero. Also export the exact time derivatives of every evolved field, using the physical metric's complete first/second spacetime jets. Lambda_t consumes a mixed second metric derivative, not a missing third metric derivative. No invented higher reference or live jets may be supplied. The eventual JSON will retain full metric jets and enough intermediate quantities for an independent consumer to reconstruct the exported Z4c fields.

Before any actual source/kernel query, independently compute full physical Christoffels, Riemann and Ricci from the physical metric jets, and the reference Christoffels from its embedding. Check flatness and g^{bc}(Gamma[g]^a_bc-Gamma[ghat]^a_bc)=0, as well as the four harmonic scalar equations for Yhat. Check the two conformal source contractions from the root derivation and the ADM lapse/shift identities on the exported states. Epsilon=0 must match the independently frozen radial reference and its Cartesian composition. These are oracle identities, not an actual tensor-kernel gate.

The fixed local tolerances and sample definitions are in plan.json. Run at 80 and 110 decimal digits; compare all saved jets and time derivatives with scale max(1,abs(x),abs(y)). Every residual is scaled by max(1,sum of magnitudes of its independently computed constituent terms); retain unscaled errors and term scales as well. Precision differences must be <=1e-55, algebraic/curvature/wave/implicit residuals <=1e-55, and scalar inversion residual <=1e-65 at both precisions. Epsilon=0 comparison with the existing binary64 reference uses unchanged2e-10 scaled tolerance. No clipping, Omega floor, damping modification, source subtraction or live physical-lapse replacement is introduced.

Subsequent helper queries require a separately accepted exact source/export interface and parent release. The planned peer helper takes a scaled physical reference connection Omega*Gammahat^a_ij and has zero reference time-connection entries; this is to be compared against the independent embedding/ADM connection, not assumed. The pure wave-map gauge has harmonic inner principal data and is distinct from current moving-puncture core gauge. No inner blend or wormhole/trumpet claim is part of this local oracle.
