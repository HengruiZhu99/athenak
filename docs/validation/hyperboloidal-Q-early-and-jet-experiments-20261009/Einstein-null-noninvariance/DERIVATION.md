# Einstein-compatible gauge jets: sigma5 non-invariance

This separate follow-up linearizes the actual C0 full Cartesian tensor kernel and frozen Q/null gauge at the outer Minkowski CMC reference. S=1,kappa_input=10,a=.5,.75,1,2; sigma0,3,5 are analytical controls. Production/storage/geometric equations are unchanged. No sigma3 candidate is admitted or built for evolution.

Let r=|x|, n=x/r, Omega=(1-r^2)/(2a), h=alpha_ref=(1+r^2)/(2a), beta_ref=-x/a. Set every geometric/P/physicalTheta field and its derivatives to the analytic reference, set delta alpha=0, and delta beta=b(r)n with b=Omega^m,m=2,3,4. A smooth cutoff equal to one in an outer collar can extend this gauge variation through the interior without changing any boundary jets. Physical vacuum ADM constraints, spatial Z and physicalTheta vanish identically initially, independent of lapse/shift. The actual kernel confirms exactly zero for all480 sampled initial constraint rows.

For m2 the initial first jet is zero: all R0, N0/N1/Qnum0/Theta0/Theta1 and shear-residue conditions in the preceding20x80 audit hold. This is genuinely Einstein-compatible initial ADM/Z4 data, unlike the earlier Theta=Omega or P=Omega^2 off-constraint controls. It does not assert a globally smooth conformal Einstein evolution under the chosen gauge.

The initial null and conformal-Q numerators are

    delta omega_n = r b/(a h)
    delta Nraw = 2r^3 b/(a^3 h^2) = 2Omega^m/a + O(Omega^(m+1))
    delta Qnum = -3r b/(a h).

Therefore Nraw/Omega^2 and Qnum/Omega are finite at the initial corner. Higher initial conformal regularity is checked rather than assumed: the actual geometric and gauge RHS give the full four-dimensional first metric derivatives, from which invariant.cpp independently constructs all Christoffels, Hessian H_ab=bar(nabla)_a bar(nabla)_b Omega and Box. With Z=0 the exact source relation is

    delta Box(Omega) = sigma delta Nraw/Omega.

The tracefree Hessian S_ab=H_ab-g4_ab Box/4 is O(Omega), and the conformal scalar curvature inferred from physical vacuum trace is

    delta Rbar = -6delta Box/Omega+12delta Nraw/Omega^2
               = (12-6sigma)delta Nraw/Omega^2,

which is initially finite. These are coordinate-component regularity checks, not a positive energy norm or bounds for curvature time derivatives. The source identity error is1.154632e-14; maximum sampled |delta S_ab/Omega| is47.999120 over all controls, bounded as Omega decreases to1e-5.

## Exact first RHS as a Cartesian field

The actual full20 RHS is independently matched to the following radial analytic field. Write b'=partial_r b and q=2(b'-b/r). Then

    alpha_t = 3h r Omega^(m-1)/a
    beta_t^i = B(r)n^i
    B = (m-2sigma)r^2 Omega^(m-1)/a^2
        +[-6/a+2r^2/(a^2 h)]Omega^m
    chi_t = 2(m-3)r Omega^(m-1)/(3a)-4Omega^m/(3r)
    gtilde_ij,t = q(n_i n_j-delta_ij/3)
    Lambda_t^i = [4m(m-1)r^2 Omega^(m-2)/(3a^2)
                 -4m Omega^(m-1)/a-8Omega^m/(3r^2)]n^i
    P_t=Theta_t=Atilde_ij,t=0.

These are complete Cartesian fields; all derivatives of h(r),r,n and Omega are retained when feeding their jets into the actual kernel. Maximum actual full20 value-field mismatch is3.552714e-15. The next actual R0 is roundoff (<=1.421086e-14), so a value/pole tangency check alone misses the failure below.

## Quadratic-null time-jet obstruction

Metric evolution gives delta G_t=2(m-1)r^3 Omega^(m-1)/a^3. Using the actual alpha/beta rates,

    delta omega_n,t = r B/(a h)+r^2 alpha_t/(a^2 h^2),
    delta Nraw_t = delta G_t+2r^2 delta omega_n,t/(a^2 h).

Consequently

    lim[delta Nraw_t/Omega^(m-1)] = 4(m+1-sigma)/a^3.

This normalization is essential. For m2,sigma5,

    (Nraw_t)_0=0,   (Nraw_t)_1=-8/a^3.

At target a=.5 this coefficient is-64. Direct derivatives of the consistent analytic first RHS jet reproduce the coefficient, rather than relying only on a small-Omega fit. Thus Nraw=O(Omega^2) is not tangent in time for this admissible initial Einstein gauge direction, even though the next R0 vanishes. A smooth spacetime Taylor ideal containing all the stated first-jet conditions cannot be invariant for sigma5 without further restrictions/changes. This does not prove finite-Q-amplitude blowup, failure of finite-Omega Einstein evolution, or a uniform instability rate: a stiff corner relaxation can invalidate interchange of time/boundary limits.

Sigma3 cancels this particular m2 coefficient. Its leading cubic passes kappa10,a.5 (K=kappa*a^2=2.5>1.5), but fails kappa5,a.5 (K1.25). Neither this radial observation nor the previously checked principal system establishes a sigma3 angular/higher-jet ideal, nonlinear closure or stability. Any such extension requires a separate gate and decision.

Additional controls keep every geometric field reference: arbitrary common rescaling delta alpha=h f,delta beta=beta_ref f with f=1+Omega/4+n_y/2 has delta Nraw=0 and delta Nraw_t=roundoff; the earlier deltaalpha=Omega,deltabeta=-Omega n Einstein witness has Nraw_t=O(Omega^2). Their successful individual corners do not remove the quadratic-shift counterexample.
