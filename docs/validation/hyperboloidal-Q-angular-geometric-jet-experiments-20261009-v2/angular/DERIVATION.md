# Gauge-only angular compatible ideal at the reference

Scope: the actual linearized C0 Cartesian kernel and frozen Q/null gauge, with exact analytic Minkowski outer CMC spatial geometry, P and physicalTheta kept reference initially. S=1,kappa_input10,a=.5,.75,1,2,sigma3/5; north and oblique(.36,-.48,.8) points. This is a gauge-only tangent calculation. Spatially perturbed Einstein geometry, nonlinear closure and stability are not established.

Write h=(1+r^2)/(2a),beta_ref=-x/a,Omega=(1-r^2)/(2a). Any linear gauge perturbation decomposes into a common rescaling plus a relative shift:

    deltaalpha=h f,
    deltabeta=beta_ref f+b_n n+b_T,   n.b_T=0.

The common part and tangential part have deltaNraw=0 identically. Quadratic-null compatibility requires b_n=Omega^2 b2+O(Omega^3). The compatible local second-jet space has dimension31: ten scalar Taylor coefficients for f,twenty for the two tangential shift components,one leading Omega^2 normal coefficient. The actual40x70 input maps span this full31-dimensional gauge space. Exact ranks refer to rational reconstruction of the analytic reference maps with common denominator216*25^6; maximum floating reconstruction discrepancy3.552714e-15. The overcomplete basis uses f multiplying {1,Omega,ny,nz,Omega^2,Omega ny,Omega nz,ny^2,ny nz,nz^2}, tangent fields Omega^m A(n)(e_j-n_j n) for m0,1,2, jxyz and six angular monomials, and Omega^2 A(n)n for the six angular monomials. Full analytic Cartesian angular derivatives are retained.

Physical ADM H/M,spatial Z and physicalTheta vanish exactly initially for every control, since geometry/P/Theta are reference and constraints are independent of lapse/shift. R0,Theta1 and shear conditions initially vanish. This is not a statement that arbitrary independent lapse/shift boundary values are admissible: the coupled null conditions have already been parameterized into the basis.

## Exact scalar transport at initial reference geometry

The actual full kernel is checked against an independent ADM/4D source derivation. For arbitrary angular f,b_n,b_T, the linear initial null response depends only on the relative normal shift. With N=deltaNraw,

    N_t=T(r) partial_r N+C(r)N
    T=-(r^4+6r^2+1)/(4ar)
    C=[r^6+(16sigma-29)r^4+15r^2-3]/[4ar^2(r^2-1)].

Tangential angular divergence terms cancel between the spatial metric trace and the preferred Box source. Common rescaling gives N_t=0 exactly. The full70-basis actual kernel verifies this scalar identity at the sampled radii and both orientations, rather than applying the radial formula without testing angular jets. The maximum raw residual is2.358908e-10, including amplified binary64 cancellation in common rescalings.

For n=N/Omega^2 the reaction is C+2T Omega_r/Omega. Sigma3 removes its pole exactly:

    n_t=T partial_r n-[(r^2+3)(3r^2-1)/(4ar^2)]n.

The reaction tends-2/a at the boundary. Equivalently

    (Nraw_t)_1=2(3-sigma)N2/a^2.

Thus sigma3 cancels the entire first null time-jet obstruction in this gauge-only compatible second-jet space. Sigma5 retains it; e.g normal Omega^2 basis has targeta.5 coefficient-64. This regular scalar equation is not a full constraint energy estimate or full geometric/nonlinear invariant ideal. It describes a derived tangent subsystem at the reference, not an imposed null/Theta falloff.

## Actual first RHS jets and pole tangency

For general gauge-only initial data, linear physical ADM kinematics give

    bargamma_ij,t = partial_i deltabeta_j+partial_j deltabeta_i
                     +[-2deltabeta.dOmega+2deltaalpha/a]delta_ij/Omega
    P_t=-Omega Delta(deltaalpha)-3x.grad(deltaalpha)/a+3deltaalpha/a
    Atilde_ij,t=-[partial_ij deltaalpha]^TF
    Theta_t=0.

chi_t and gtilde_t follow trace/determinant completion, and Lambda_t is the contracted connection time derivative. The actual full20 geometric RHS matches these complete fields to7.556622e-12. Their first Cartesian derivatives are evaluated in factored form for common/tangential/normal families, avoiding unused division cancellations in the analytic oracle. The P gradient at the boundary comes from differentiating the actual matched field:

    partial_i P_t = [n_i Delta(deltaalpha)-3n^j partial_ij deltaalpha]/a.

This supplies actual first-RHS jets, not independent choices of trace gradients. The full actual kernel applied to those jets gives next R0 maximum5.566730e-9 and leading H/M/Z/Theta maximum6.926193e-10. These checks include lapse/shift rates from the actual Q/null helper. They confirm first R0/shear/Theta time tangency on this gauge-only reference family for both sigma controls; sigma5 nevertheless fails N1 time tangency as above.

## Numerical precision and limitations

1120 controls and13440 sampled actual-kernel points cover four a,two sigma,two orientations,seventy spanning fields and twelve radial samples each. Four-point one-sided extrapolation at h=.001,.0005,.00025 measures N1_t. Maximum discrepancy from the exact expected map is 1.276410e-7 at h.001,6.062952e-7 at h.0005 and1.954123e-6 at h.00025; cancellation grows as Omega decreases. These are retained errors, not convergent FD order evidence or an asserted1e-14 zero. The exact transport identity and its pole cancellation are separately symbolic, and the raw actual kernel residual is reported separately. Release and ASan/UBSan Debug are matched.

The map covers all compatible gauge second jets at fixed reference spatial geometry. It does not cover physical gravitational data or general spatial diffeomorphism perturbations of the ADM geometry. Those require a separate actual geometric jet gate. Sigma3 is not admitted for any native/global evolution; prior frozen sigma5 negative controls remain unchanged.
