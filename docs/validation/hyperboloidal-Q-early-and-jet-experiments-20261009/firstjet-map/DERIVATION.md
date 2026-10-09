# Actual Q/null first-jet singular map

Scope: linearization of the actual Cartesian C0 kernel and frozen Q/null gauge at the analytic Minkowski outer CMC reference, S=1, kappa_input=10, sigma=5, a=.5,.75,1,2. The inner alpha blend and either earlier feedback continuation have identical W=1 boundary jets. This is a necessary Taylor compatibility analysis, not a nonlinear closure, subsidiary stability, boundary implementation or amplitude blowup proof.

At the north point r=1 write c=delta chi, h_ij=delta gtilde_ij, A_ij=delta Atilde_ij, p=delta P, theta=delta Theta_phys, d_alpha=delta alpha, b=delta beta_n, x=c-h_nn and s=d_alpha+b. Let d_i=delta Lambda_i-delta contractedGamma_tilde_i, and H_ij=h_ij+delta Gamma_bar^n_ij with bar gamma=gtilde/chi. All derivatives below are Cartesian, including the nonfree angular derivatives of n_i. Define Htf by the Euclidean reference tracefree projection and

    SA_ij = 2/a^2 { Htf_ij - 1/2[n_i d_j+n_j d_i-(2/3)delta_ij d_n] - A_ij }.

The singular numerator R0 is:

    R0_alpha = (-p+3s)/a^2
    R0_chi = 2(p+2theta-3s)/(3a)
    R0_P = -2(p+2theta)/a^2 -3x/a^3 +10theta
    R0_Theta = -2p/a^2 -(1/a^2+20)theta -3x/a^3
    R0_beta_n = -5[x+2as]/a^3,   R0_beta_tangent=0
    R0_metric=0,   R0_A=SA
    R0_Lambda_i = -(10-2/a^2)d_i -2(2 partial_i p+partial_i theta)/(3a)+4A_in/a^2.

jet_map.cpp checks this complete formula on 320 actual dual basis columns (four radii). Each local first jet has 80 coordinates: the 20 primitive coefficients multiplying {1,Omega,n_y,n_z}. Nonlinear det(gtilde)=1 and tracefree A are completed before calling the actual kernel; full analytic Cartesian first/second derivatives of these basis functions are retained. The singular numerator itself depends only on first jets. The resulting 20x80 map has rank11 and nullity69, while its value-only 20x20 restriction has rank11 and rank(M^2)=11. Exact rank statements refer to rational reconstruction of these analytic-reference matrices, with maximum floating reconstruction error3.552714e-15.

## Boundary value freedoms

The linear null norm and conformal-Q numerator are

    N0 = x/a^2 +2s/a,     Qnum0 = p-3s.

Thus R0_alpha=-Qnum0/a^2 and R0_beta_n=-5N0/a. The remaining scalar rows reduce to

    R0_chi = 2(Qnum0+2theta)/(3a)
    R0_P = -2Qnum0/a^2 -3N0/a +(10-4/a^2)theta
    R0_Theta = -2Qnum0/a^2 -3N0/a -(1/a^2+20)theta.

N0=Qnum0=theta0=0 cancels this complete scalar/gauge value block. Unlike the old physical-P/spatialnorm value conditions, it does not force independent zero lapse and normal-shift deviations. Their sum is tied to geometry and physical P: x=-2as and p=3s. Tangential beta values have no leading pole. These are coupled necessary conditions, not permission to prescribe alpha and beta independently at scri. N1 is separately required for quadratic null regularity and is not inferred from the value pole kernel.

## Leading constraints are insufficient

The actual Hamiltonian leading coefficient satisfies

    H0 = -4 Qnum0/a -8 theta0/a -6 N0.

The actual momentum/Z identity is

    M0_i -(10a-2/a) Z0_i +partial_i theta0 = (a/2) R0_Lambda_i.

Here Z0 is the physical spatial covector, and partial_n theta=-Theta1/a. A pointwise Theta0=0 does not kill its tangential derivatives. If Theta0 vanishes as a boundary field and Theta1=0, M0=Z0=0 imply R0_Lambda=0. These are consequences of the displayed identity; no stronger Theta falloff is imposed as a shortcut.

The condition matrix E in actual.json records H0,M0(3),Z0(3),Theta0,N0,N1,Qnum0, tangential gradients of N0/Qnum0/Theta0, H1, Theta1 and five SA components. H1 is verified independently by fourth-order one-sided extrapolation of the actual Hamiltonian constraint. The first11 conditions have rank10; adjoining R0 raises it to18. Including the six tangential identities and H1 gives rank17; adjoining R0 raises it to20. Adding just Theta1 and the two tangential tracefree shear residues SA_yy,SA_yz gives rank20 and contains every row of R0. The other shear relations then follow in this reference linear first-jet space. This exact rowspace statement is not a general nonlinear boundary theorem.

Two actual-kernel counterexamples make the missing conditions concrete:

* Theta=Omega and p=-2Omega, all other deviations zero. Physical delta K=p+2Theta is zero, so the linear physical ADM constraints and Z vanish identically, including H1. Every listed leading null/Q/Theta coefficient and tangential identity vanishes, but Theta1=1 and R0_Lambda_n=-2/a^2. This is off the Einstein sector because physical Theta is nonzero inside.
* At the north point Ayy=1,Azz=-1, with first angular jets Axy=-n_y,Axz=+n_z and all other deviations zero. These are the local jets of a tangential tracefree shear. All leading listed constraints/null data, H1 and Theta1 vanish, but R0_Ayy=-2/a^2. This is a local leading-constraint counterexample, not a full Einstein initial data construction on the sphere; higher constraints need separate checking.

The strong Einstein gauge witness delta alpha=Omega, delta beta=-Omega n has identically zero initial physical H/M/Z/Theta and zero R0. Its earlier checked Q/sigma null time corner remains a single witness rather than proof that general compatible jets form an invariant set.

## One next level from the actual full kernel

Take p=Omega^2 and other deviations zero. Its first jet and R0 vanish, as do all E first-jet conditions. It is not exact Einstein data: its higher Hamiltonian and momentum coefficients are nonzero. Its first actual assembled RHS, as an exact Cartesian field, is

    alpha_t = -alpha_ref(r)^2 Omega
    chi_t = 2alpha_ref(r) Omega/3
    P_t = -2Omega^2/a
    Theta_t = -2alpha_ref(r) Omega/a
    Lambda_t^i = 8alpha_ref(r) x^i/(3a),

with other components zero and alpha_ref(r)=(1+r^2)/(2a). The full varying alpha_ref is used when differentiating; it is not replaced by constant alpha_scri=1/a. The actual first RHS agrees with this field to1.776357e-15 at sampled positive Omega. Feeding its consistent Cartesian jet into the actual kernel produces

    R0_A_nn = -8/(3a^4),
    R0_Lambda_n = (12-80a^2)/(3a^4),
    (Nraw_t)_1 = -4/(3a^3).

Thus first-jet singular cancellation is not invariant on arbitrary off-constraint higher jets. This control alone says nothing decisive about exact Einstein-compatible higher jets, which require a separate actual-kernel calculation.

No boundary ghost continuation, nonlinear constraint manifold or evolution-preserved finite-Q amplitude claim follows. In particular the earlier finite-Q derivative counterexample is not evidence of finite-Q amplitude blowup in the stiff stable layer.
