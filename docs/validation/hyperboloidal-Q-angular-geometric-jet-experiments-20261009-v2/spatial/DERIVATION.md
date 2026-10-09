# Einstein spatial pullbacks and the Q/null first time jet

This is a linear, local, actual Cartesian full20 audit in the outer Minkowski CMC region. Geometric storage/evolution remains physical P; the existing frozen Q/null helper is used unchanged with sigma3/5, C0 and kappa_input10. S=1,a=.5,.75,1,2. No native/global propagation or new gauge admission is made.

Let Omega=(1-r^2)/(2a), h=(1+r^2)/(2a), beta_ref=-x/a. A time-independent spatial diffeomorphism with infinitesimal generator xi gives exact linear Einstein initial ADM data, including lapse and shift pulled back from the same stationary Minkowski foliation. With prescribed Omega held fixed,

    delta bargamma_ij=partial_i xi_j+partial_j xi_i-2(xi.dOmega)/Omega delta_ij,
    delta chi=-2 div(xi)/3+2(xi.dOmega)/Omega,
    delta gtilde_ij=partial_i xi_j+partial_j xi_i-2 div(xi) delta_ij/3,
    delta Lambda_i=Delta xi_i+(1/3)partial_i div(xi),
    delta P=delta Theta=delta Atilde=0,
    delta alpha=xi.dh-h(xi.dOmega)/Omega,
    delta beta=(x.grad(xi)-xi)/a.

The physical Kij is -gamma_ij/a, so H/M vanish exactly, and the consistent metric connection gives spatial Z=0. Theta vanishes identically. These are genuine Einstein/Z4 initial data, rather than data with only leading ADM coefficients zero. A smooth radial cutoff in the outer region can extend the local generator into the exact full Minkowski reference without changing the displayed outer jets. No black-hole or reference-RHS subtraction is involved.

The code generates complete analytic Cartesian jets through second derivatives for xi=Omega^m X(x)e_j, m1/2/3/4, jxyz and the ten Cartesian monomials of degree at most2. The alpha expression is factored as Omega^(m-1)x_j X/a^2. Derivatives retain Omega as a symbol with total derivative partial_i=explicit_partial_i-(x_i/a)partial_Omega. This prevents cancellation from expanded high powers of r^2-1. The actual geometric first RHS should vanish for this complete stationary pullback; that necessity is checked in the actual kernel, not assumed instead of evaluating it.

## Independent stationary four-dimensional source identity

Write w=xi.dOmega and f=w/Omega. The complete stationary conformal four-metric perturbation is Lie_xi(gbar)-2f gbar. Scalar covariance and the four-dimensional conformal wave transformation give

    delta Box_stationary=xi.d(Box_ref)-Box_ref(w)+2f Box_ref-2 grad(f).grad(Omega),
    delta Nraw=xi.dNraw_ref-2 grad(w).grad(Omega)+2f Nraw_ref.

These identities retain angular derivatives. In the actual preferred gauge, with Z=Theta=0 initially, Box(Omega)=Box_ref(Omega)+sigma deltaNraw/Omega. Because the physical ADM geometric rates of this stationary pullback vanish, the gauge response therefore obeys

    delta omega_n,t=h[delta Box_stationary-sigma deltaNraw/Omega],
    delta Nraw_t=-2omega_n,ref h[delta Box_stationary-sigma deltaNraw/Omega]
                =(2r^2/a^2)[delta Box_stationary-sigma deltaNraw/Omega].

The sign also follows from an independent radial four-metric volume/divergence computation in check_diffeo.py. That script derives the variation of gbar^rr and sqrt(-gbar)=h r^2 and differentiates their product; it does not assign the boundary rates as axioms.

## A larger compatible Einstein family fails sigma3 tangency

For m1, put Y=n_j X(n) on the unit sphere. Direct expansion of the preceding four-dimensional identities gives

    delta Nraw_2=-2Y/a,
    delta Box_stationary,1=[Delta_S Y-4Y]/a,
    (delta Nraw_t)_1=(2/a^3)[Delta_S Y+(2sigma-4)Y].

For xi=Omega*x (the sum of three generated Cartesian columns), Y=1. Equivalently xi=Omega*n has the same leading boundary data. Both have

    delta Nraw_2=-2/a,
    delta Box_stationary,1=-4/a,
    (delta Nraw_t)_1=4(sigma-2)/a^3.

Thus sigma3 gives +32 at a.5, and sigma5 gives +96. Initially all physical H/M/Z/Theta constraints vanish exactly; N0=N1=Qnum0=Theta1=0 and the full actual R0/shear pole conditions hold. The actual first RHS leaves R0 tangent but creates a nonzero N1 time coefficient. The initial full four-dimensional Christoffel/Hessian calculation satisfies the preferred Box identity and has finite sampled tracefree Hessian/Omega and conformal scalar curvature. This is not an omitted initial ADM constraint, Theta residue or conformal shear singularity.

The previous gauge-only beta=Omega^2*n witness instead requires sigma3: N1_t=4(3-sigma)/a^3. Consequently no single constant sigma cancels both independent directions in this larger linear smooth Taylor ideal. Additional restrictions or a different derived source might change that conclusion; neither is imposed here. In particular, choosing N2 or Theta falloffs by hand would not establish invariant closure or retention of physical radiative data.

For m2+ the spatial-pullback family has N1_t=0 for both sigma controls. The initial narrower pilot therefore could not detect the m1 geometric obstruction. The full angular m1 map and the radial counterexample are distinguished from those higher-order controls.

## Scope and numerical precision

The final receipt contains120 analytic pullback fields,1920 parameter/orientation controls and23040 actual-kernel points. Release and ASan/UBSan Debug outputs are matched. Actual boundary N0/N1/Q0, constraints and all20 R0 residues are measured; next R0 uses actual gauge first-RHS values and the independently checked zero stationary geometric first-RHS jets. The latter derivative oracle is justified by the complete stationary physical pullback, rather than arbitrary independent geometric time jets.

Normalized N1 rates are obtained using four-point one-sided extrapolation at h=.001,.0005,.00025. SmallOmega binary64 cancellation is retained, and this is not claimed as a convergent FD-order study. Initial four-dimensional Box checks also report their amplified normalization error separately. REPORT.md/check-report.json record the final measured values.

This is failure of smooth-in-time quadratic-null Taylor compatibility on a genuine initial Einstein subset. It does not prove finiteOmega amplitude blowup, a global continuum spectrum, failure of every sigma3 finiteOmega implementation, or a full nonlinear hierarchy theorem. Fast singular relaxation can invalidate a smooth corner expansion without producing an order-one constraint amplitude. No sigma3 evolution is admitted by this gate.
