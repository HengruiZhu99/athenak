# A fixed conformal scri frame is not an invariant restriction of the current gauge

This is a fresh local linear audit of the frozen C0 Q/null helper with sigma3, eta_outer1, S=1,a=.5,.75,1,2. P storage/evolution is unchanged. All prior frozen sources stay byte-identical. No boundary prescription, native/global evolution, new source, or sigma3 admission is introduced.

## Geometric two-metric and tangential shift

Let q_AB be the pullback of the Penrose spatial metric bargamma=gtilde/chi to Omega=0 in fixed angular coordinates. Put nu=|D Omega|_bargamma>0 and s=-D Omega/nu, the outward unit spatial normal. On the future null branch omega_n=-nu, the shift normal component is beta_s=-alpha. The geometric tangential shift is

    Y=beta+alpha*s.

It is not the coordinate tangential components of beta when bargamma_nA is nonzero. At the analytic reference,

    delta Y_A=delta beta_A-delta bargamma_nA/a.

The actual geometric equations imply bargamma_t=Lie_beta(bargamma)-2alpha*Kbar. With k_AB the second fundamental form of the spatial cut, the Einstein tracefree conformal-Hessian condition gives

    Hess(Omega)_AB=-nu(k_AB+Kbar_AB)=(Box(Omega)/4)q_AB at scri.

Equivalently the null generator is ell=(nu/alpha)(partial_t-Y), and Lie_ell(q)=2Hess(Omega)|AB. Therefore

    q_AB,t=Lie_Y(q)_AB+[alpha Box0/(2nu)]q_AB.

For Einstein data with Nraw=O(Omega^2), spatialZ/Theta zero and the preferred source Box(Omega)=O(Omega), Box0=0. Then q_t=Lie_Y q. This is a consequence of compatibility, not a boundary value imposed on all metric components.

A fixed coordinate two-metric requires Y to be Killing. A fixed coordinate conformal class only requires Y to be conformal Killing; its scale then changes with div_q Y. Fixing only the area element requires div_q Y=0 and does not fix tracefree shape. Modulo boundary diffeomorphisms, q_t=Lie_Y q transports the same geometry even for a general Y; that is different from fixed component values in the existing coordinates. Changing to angles transported by Y changes the coordinate gauge and is not silently an invariant subset of the current prescribed shift equations.

## Why conformal class alone does not remove the prior obstruction

The prior exact Einstein spatial pullback xi=Omega X e_j has

    delta q_AB=(2Yscalar/a)q_AB, Yscalar=n_j X(n).

The entire m1 normal-rescaling obstruction changes only conformal scale at the boundary. Fixing conformal class retains it. Fixing q or its scale to the reference excludes that specific direction, but exclusion of one counterexample proves no invariant closure. Such a restriction also does not imply all metric deviations are O(Omega): bargamma_nn,bargamma_nA and compatible gauge values can remain finite.

## Conditional intrinsic roundness is a different restriction

In two dimensions q_t=Lie_Y q+lambda q implies R[q]_t=Lie_Y R[q]-lambda R[q]-Delta_q lambda, lambda=alpha Box0/(2nu). When the full Einstein/null hierarchy already guarantees Box0=0, a constant unit-curvature condition R[q]=2 is transported invariantly for arbitrary Y. This permits changing coordinate components of a round metric. The gauge-only counterexample below is precisely such a pure angular coordinate drift: its q_tt is a Lie derivative and deltaR[q]=0. It does not refute intrinsic roundness.

For the prior m1 scale perturbation deltaq=2psi q, psi=Yscalar/a, the linear intrinsic curvature is deltaR[q]=-2(Delta_S+2)psi. At sigma3 the spatial-pullback obstruction is N1_t=-(deltaR[q])/a^2. Thus fixing intrinsic unit curvature removes that particular obstruction while retaining its degree1 conformal-frame kernel. This is a promising conditional compatibility observation, not a proof of the full coupled invariant ideal: using Box0=0 to prove roundness transport does not independently prove that N1=0, shear, Einstein constraints and their higher jets persist. Physical radiative Einstein data beyond the pure-diffeomorphism subset have not been tested here.

Adding R[q] directly to the beta gauge would introduce second spatial derivatives and change the principal order. No such source is prototyped, and a metric-only value function cannot be assumed to implement this differential correction. No initial or boundary Dirichlet condition is adopted.

## Actual gauge-only counterexample with fixed initial q and Y

Keep exact reference ADM geometry,P,Theta and perturb only

    delta beta=Omega*T(x), x.T=0.

Use T=z(r^2 e_z-z*x), whose restriction is n_z(e_z-n_z*n), and the Killing control T=e_z cross x. The first is proportional to the gradient of a degree2 sphere harmonic and has a genuine tracefree angular Lie derivative. Both have initial q=qref,Y0=0,N0=N1=Qnum0=Theta1=0, exact physical H/M/Z/Theta constraints zero, and all actual R0/shear pole residues zero.

For homogeneous polynomial T of degree d, the complete actual first RHS fields are

    bargamma_F=sym d(Omega*T), chi_F=-2div(Omega*T)/3,
    gtilde_F=bargamma_F+chi_F I,
    Lambda_F=Delta(Omega*T)+grad div(Omega*T)/3,
    beta_F=[r^2/a^2-((d+1)/a+eta_outer)Omega]T,
    alpha_F=P_F=Theta_F=Atilde_F=0.

These fields are independently derived from the actual equations and matched against the complete actual full20 RHS. Their full analytic Cartesian jets are generated, then supplied to the actual kernel for the second response. They are not independent guesses of first-RHS derivatives.

At scri, beta_F,T=T/a^2 and bargamma_F,nT=-T/a. Including the moving spatial normal gives

    Y_t=2T/a^2,
    q_AB,tt=Lie_(2T/a^2)(qref)_AB.

Thus fixed initial q and Y do not suffice. At a.5,n=(1,0,0), the non-Killing control gives q_tt=diag(0,16) in the y,z tangent basis, with tracefree part diag(-8,8). The Killing control gives zero q_tt. The same actual first-RHS jets preserve N1 to roundoff: this is a separate tangential-frame obstruction even after the normal sigma3 gauge-only condition passes. The null feedback is radial and cannot change this tangent sector; eta_outer1 is used explicitly, and changing eta cannot remove this gauge-only boundary rate because delta beta0=0.

## Remaining tangential time-jet condition

At the reference, using fixed Cartesian n/tangent projections at a point, the actual tangential shift plus geometric normal evolution gives

    Y_A,t=(Lambda_A+.5 partial_A chi+partial_n h_nA+2 Atilde_nA)/a^2
          -(2/a)partial_n beta_A-(1/a)partial_A(alpha+beta_n)
          -(eta_outer+1/a)beta_A,

where all terms denote perturbations and h=delta bargamma. This formula retains the evolved normal frame. A fixed q with Y=0 additionally requires Y_t to be Killing, followed by compatible higher time derivatives; equivalently its tracefree sphere symmetrized gradient and divergence must vanish. This is a derived coupled first-jet condition, not independent Dirichlet values for alpha,beta or metric. The current gauge does not impose it.

A separate read-only postprocess of the prior frozen Einstein spatial-pullback data uses xi=Omega(r^2 e_y-y*x). It also has initial q=qref,Y0=0, but gives Y_t=(eta_outer*a-1)T/a^3. The default eta_outer1 gives -4T at a.5. That family is distinct from the gauge-only Omega*T family; the latter gives +2T/a^2 and is independent of eta at the boundary. Setting eta=1/a could cancel the former necessary condition only, not the latter or a full hierarchy.

## Radiation and scope

Fixing the intrinsic leading q_AB is a possible conformal/coordinate frame choice, not itself a requirement of no radiation. The local shear condition allows Kbar_AB^TF=-k_AB^TF and hence finite compatible Atilde_AB determined by normal derivatives of q; it does not require A or Lambda to vanish as Omega. In regular conformal Einstein characteristic formulations, independent rescaled-Weyl data and corner frame data are explicitly separated; see Hilditch, Valiente Kroon and Zhao, Section4.1/Lemma3, https://arxiv.org/pdf/2006.13757. This supports the distinction between a boundary frame and radiation data, but does not prove that the current C0 Q gauge with an added frame restriction retains every radiative solution.

No actual radiative TT family or full invariant constrained Taylor ideal is established here. A justified fixed-frame formulation would still need an evolution/characteristic gauge that preserves the coupled normal, shear, tangential and higher time-jet conditions while allowing the radiative free data. The present counterexample rules out declaring fixed initial q/Y sufficient, not every possible conformal-frame gauge. No ad hoc stronger Theta or metric falloff is imposed.
