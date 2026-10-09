# Exact tensor C_Z4c additions in physical-P storage (scratch)

Use gtilde for the determinant-one spatial metric, Atilde for its tracefree
curvature, Ztilde^i=gtilde^ij Z_j, theta for physical Theta, and K=P+2theta
for the physical trace. Omega is fixed in coordinate time. The physical lapse
is A=alpha/Omega and physical gamma_ij=gtilde_ij/(Omega² chi).
The runtime damping normalization remains kappa1=kappa_input/alpha.

Mechanical Appendix B conversion uses paperTheta=theta/Omega,
paperK=(P-3w)/Omega, paperZ^i=chi Ztilde^i and
w=-beta.dOmega/alpha. Holding the gauge RHS fixed makes delta w_t=0,
so delta P_t=Omega delta paperK_t. This yields the complete C-dependent
additions (multiply every displayed term by C_Z4c):

    delta chi_t = delta gtilde_ij,t = 0,
    delta Atilde_ij,t = -2 alpha Atilde_ij theta/Omega,
    delta P_t = 2 chi Ztilde^i (Omega d_i alpha-alpha d_i Omega),
    delta theta_t = -Omega chi Ztilde^i d_i alpha
                    -alpha Omega Ztilde^i d_i chi/2
                    -alpha theta(K-3w)/Omega,
    delta Lambda^i_t = -2 theta gtilde^ij d_j alpha/Omega
                       +2 alpha theta gtilde^ij d_j Omega/Omega².

These are the full tensor terms, with neither a spherical reduction nor a
reference RHS subtraction. No chi multiplies the two Lambda terms. They
vanish exactly for theta=0 and Lambda=contractedGamma, and C=0 returns zero
exactly. Alpha and beta gauge equations, physical-P stabilization and
kappa_input/alpha are unchanged. The generic scratch header represents the
regular, simple-pole and double-pole numerators separately. It rejects
nonpositive Omega in assembly rather than flooring or extending the PDE.

The physical extrinsic-curvature equation becomes

    Kphys_ij,t = ADM_ij
       + A(Dphys_i Z_j+Dphys_j Z_i-2theta Kphys_ij)
       - A kappa1(1+kappa2)theta gamma_ij.

The physical Theta equation becomes

    theta_t = beta.dtheta + A(H/2+div_phys Z-K theta)
               - Zphys^i d_i A-A kappa1(2+kappa2)theta.

The nonlinear fulltensor probe constructs gamma and Kphys jets by exact
products from the evolved fields, independently computes the physical Ricci,
lapse Hessian, Lie derivatives and Z covariant derivatives, and compares
against the actual C0 kernel plus additions. This directly verifies the
physical identities rather than presuming covariance from the paper's label.
At tiny Omega individual physical ADM terms diverge, so the receipt retains
both absolute cancellation residuals and termwise normalized residuals.

## Printed Appendix A/B offconstraint distinction

The mechanical Appendix-B C1 additions do not alter the actual connection
shift term -contractedGamma^j d_j beta^i. Consequently its spatial covector
identity remains

    Z_i,t = Lie_beta Z_i
      + A(M_i+d_i theta-2Kphys_i^j Z_j-kappa1 Z_i)-theta d_i A
      + gtilde_ij Ztilde^k d_k beta^j.

A separate derived covariant completion is

    delta Lambda^i_t = -2 Ztilde^j d_j beta^i,

which replaces -contractedGamma.d beta by -Lambda.d beta and cancels the last
term exactly. This is NOT a printed C-dependent Appendix-B term. The actual
metric/Gamma time derivative is independently differentiated in the probe.

The distinction already occurs for Omega=chi=alpha=1, constant gtilde=I,
Atilde=P=theta=0, Z=(1,2,3), and d_x beta^x=1 (other shift gradients zero).
Every mechanical C1 addition is zero, but Z_t-Lie_beta Z-(damping)=(1,0,0).
The completion restores the expected covector identity. The printed Appendix A
uses partial_perp=partial_t-Lie_beta and lacks this extra term, whereas printed
Appendix B includes the independent +2 Z.d beta term. This is a demonstrated
offconstraint distinction, without inferring author intent or treating the
printed Appendix-B system as fully covariant.

## Principal and regularity scope

All additions are lower order in the interior first-order-in-time,
second-order-in-space reduction. The actual 360-case full20 principal
extraction, including physical-P and legacy flags, oblique SPD metrics and
harmonic transition/endpoints, retains the complete canonical basis. Terms
containing Ztilde are products of a connection/first metric derivative with a
first lapse/chi derivative, not new second derivatives. The covariance repair
is also lower order. This interior result does not supply a uniform boundary
symmetrizer.

The double-pole coefficient 2alpha theta gtilde^ij d_jOmega is generally
nonzero for bounded nonzero physical theta and dOmega!=0. No falloff for live
theta is assumed. It cannot be silently put into the existing regular+pole/Omega
API with a regular numerator. A large matrix entry alone does not determine an
Omega² timestep; eigenvalues, nonnormal propagators and the actual RK polynomial
must be checked before any finite-Omega native prototype. Those later spectral
and integration gates are separate from this frozen tensor identity audit.

Primary source: https://arxiv.org/html/1412.3827v2#A2, tensor Appendix B
Eqs46c-f, compared to the definitions/equations in Appendix A. The independent
covector counterexample was also checked by the literature agent. No native
runtime, gauge default, Bianchi closure, nonlinear scri manifold or BH evolution
has been changed or established.
