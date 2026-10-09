# Independent initial Einstein gauge-only null evolution

This derivation uses the physical ADM metric equation and the prescribed
four-dimensional Box source. It does not call the Z4c kernel and does not
claim a closed evolution equation on general perturbed geometry. It evaluates
the linear initial rate on exactly reference spatial/extrinsic geometry,
P and Theta, with arbitrary initial lapse/shift perturbations. These initial
physical ADM/Z/Theta constraints vanish identically. The outer S=1 reference
has bar gamma=I, Kphys_ij=-gamma_phys_ij/a, h=(1+r^2)/(2a),
beta_ref=-r n/a and Omega=(1-r^2)/(2a).

Set f=delta alpha and b=delta beta. Since Omega_t=0, ADM gives

    delta(bar gamma_t)_ij = partial_i b_j+partial_j b_i
                           +2(f/a-b.dOmega)delta_ij/Omega.

Let Bh=beta_ref.dOmega and Nraw=g4inv(dOmega,dOmega). The definition of
Box in divergence form implies

    alpha^2 Box(Omega)
      = B_t-B alpha_t/alpha+(B/2)tr(bar gamma^-1 bar gamma_t)
        +alpha/sqrt(det bar gamma) partial_i
          [alpha sqrt(det bar gamma) g4inv^ij Omega_j].

At the initially reference spatial metric, write the last variation as dJ.
Using delta Box=sigma deltaN/Omega and the stationary reference yields

    deltaN_t = -Omega_i Omega_j delta(bar gamma_t)_ij
               -2Bh deltaBox -4Bh f Box_ref/h
               +(Bh^2/h^2)delta tr(bar gamma_t)+(2Bh/h^2)dJ.

The script evaluates this expression exactly in SymPy, first for arbitrary
radial f=F(r), b=B(r)n and then for explicit powers of Omega. It proves

    deltaN_t = T(r) partial_r(deltaN)+C(r)deltaN,
    T=-(r^4+6r^2+1)/(4ar),
    C=(r^6+16r^4 sigma-29r^4+15r^2-3)/(4ar^2(r^2-1)).

This is an identity for the initial gauge-only tangent subspace. It is not
a subsidiary equation for general evolving Z4c data. The formulas concern
the outer collar; no origin extension of the 1/r expressions is asserted.

For n=deltaN/Omega^2 the reaction is C+2T Omega'/Omega. At sigma=3 it reduces
exactly to -(r^2+3)(3r^2-1)/(4ar^2), with boundary value -2/a. For other
sigma the weighted reaction has a pole. Accordingly initial quadratic null
data have (deltaN_t)_1=2(3-sigma)N2/a^2 in this gauge-only subspace.
For b=Omega^m n, f=0 the leading rate is
4(m+1-sigma)Omega^(m-1)/a^3. At m=2,sigma=5 this gives -8/a^3.
For f=1,b=-n, N2=-a and (deltaN_t)_1=2(sigma-3)/a.
For b=(beta_ref/h)f the initial deltaN and its initial rate vanish exactly.

Angular scalar amplitudes in f and b_r can be treated as parameters in this
initial identity. Tangential b_T cancels: its contribution to the metric
trace term is 2Bh^2 div_S(b_T)/(h^2 r), while its contribution through dJ is
the negative of that expression. This cancellation is an analytical statement
here, not a new actual-kernel angular basis gate. Spatially perturbed Einstein
geometry and the time invariance of the full compatible Taylor ideal remain
separate questions.

The independent pole Routh condition at sigma=3 is K=kappa_input a^2>3/2.
Thus the target a=.5,kappa_input10 satisfies it; the earlier kappa5,a.5
control does not. No sigma-three native or global candidate is admitted by
this derivation, and no finite-Q/null amplitude blowup is inferred from an
initial smooth-in-time Taylor obstruction or stiff temporal boundary layer.

The actual sigma-five kernel control is separately frozen at index
4cfbc8ed743c46787f617c33fdbe9875ec092fd117e82f1daddf5e165f2ef29a.
That gate, rather than this symbolic script, supplies independent checks of
initial finite conformal Hessian/Box regularity and actual assembled RHS jets.
