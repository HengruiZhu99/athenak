# Linear outer-CMC scri Taylor compatibility

The scope is the actual Cartesian C0 tensor RHS plus the physical-P lapse and spatial-norm shift gauge, S=1, kappa_input=10, a=.5,.75,1,2. Baseline kappa2=0 and the live candidate's V=1 outer coefficient are both evaluated. Nothing here changes production equations, imposes a field falloff, evolves native arrays, or proves a closed scri PDE. All fields and jets below are first variations about the stationary CMC reference.

Write delta u=u0(n)+Omega u1(n)+Omega^2 u2(n)+..., where u1 is the Taylor coefficient (not the coordinate-r derivative). At the unit scri sphere,

```
alpha_hat=1/a, beta_hat=-n/a, w_hat=-1/a,
Omega_i=-n_i/a, Omega_ij=-delta_ij/a,
partial_i delta u = -n_i u1/a + tangential_i u0.
```

The compiled map B has shape20x80: columns are the20 Cartesian components of u0, u1, derivative_ny u0, derivative_nz u0 at n=(1,0,0). These angular first jets are coordinate derivatives of Cartesian components, not independently rotating tensor-frame components. The map contains all actual pole numerators, including metric-connection and P/Theta derivative terms. Its rank is15 in all eight cases. Five semisimple zero modes of its value-only block do not determine this first-jet map.

An explicit equivalent expression follows. Let h=delta gtilde (tracefree), c=delta chi, p=delta P, theta=delta Theta_phys, da=delta alpha, b=delta beta; all in these formulas are boundary values unless differentiated. Define

```
x=c-h_nn,
d_i=delta Lambda_i-delta Gamma_tilde_i,
H_ij=h_ij+(delta Gamma_bar^n_ij)^TF,
Gamma_bar^n_ij=n_k Gamma[gtilde/chi]^k_ij,
eta=1.5/a^2, C=1/(3a), kappa=kappa_input,
k2star=0 (baseline) or 2/(kappa*a^2)-1 (live outer reference).
```

TF and index raising in these linear formulas use the flat reference. The full leading pole residue R0 is

```
R0_alpha = (-p-3da+b_n)/a^2,
R0_chi   = 2(p+2theta-3da-3b_n)/(3a),
R0_P     = -2(p+2theta)/a^2-3x/a^3+kappa(1-k2star)theta,
R0_Theta = -2p/a^2-[1/a^2+kappa(2+k2star)]theta-3x/a^3,
R0_beta  = -eta [b+C n x],
R0_g     = 0,
R0_A     = 2/a^2 [H-(d n+n d)^TF/2-A],
R0_Lambda_i = -(kappa-2/a^2)d_i
              -2(2 partial_i p+partial_i theta)/(3a)
              +4A_in/a^2.
```

The actual640-column comparison verifies this entire expression, including arbitrary value, normal and angular seeds, to3.553e-15. All terms involving first derivatives of the compactification/reference and Cartesian connection are retained.

For the audited kappa*a^2>1, R0=0 necessarily implies

```
da0=p0=theta0=b0=0, c0=h_nn0.
```

These are stronger than finite Q alone. They imply delta(P-3w)_0=0 and delta Nraw_0=0, where Nraw=chi*gtildeinv(Omega,Omega)-w^2. Quadratic-null regularity additionally requires Nraw1=0 and is not silently included in R0=0. Since these boundary-value equalities hold over the sphere, their angular derivatives are not free:

```
D_A c0 = n^i n^j D_A h0_ij + 2h0_nA,
D_A da0=D_A p0=D_A theta0=D_A b0=0.
```

Here D_A differentiates Cartesian tensor components in the first term; the second term is the derivative of the normal. Rotating radial A and Lambda have their corresponding nonzero angular derivatives in the actual witness. No A/Lambda=O(Omega) restriction is made.

On these necessary values, the remaining R0 equations solve finite boundary A and Lambda in terms of metric first jets and P1/Theta1:

```
d_A = 4H_nA/(kappa*a^2),
d_n = [12H_nn+4P1+2Theta1]/(3kappa*a^2+2),
A_ij = H_ij-(d_i n_j+n_i d_j)^TF/2,
Lambda0 = Gamma_tilde0+d.
```

The full kernel verifies160 random first-jet cases, with nonfree angular chi jets, to4.441e-16 in both Release and Debug ASan/UBSan.

For a continuously differentiable spacetime extension with Theta=Omega*tau, another necessary initial-corner condition is

```
Theta_t0 = (2/a)delta Lap_bar(Omega) - (2/a^2)delta Q
           -kappa(2+k2star)Theta1 -(3/a)Nraw1 = 0,
delta Q = P1-3(alpha1+beta_n1),
Nraw1 = (chi1-h_nn1)/a^2+2(alpha1+beta_n1)/a.
```

The last term must be retained for general R0-compatible first jets. It vanishes only when quadratic-null compatibility is separately imposed. A first version of this formula omitted it and failed a random actual-kernel test by7.16784; the exact failed source/output is preserved. The corrected general limit agrees with the actual kernel to3.842e-11. Smooth finite Q likewise requires (P_t-3w_t)_0=0. Preserving quadratic-null behavior requires the corresponding Nraw time coefficients to vanish.

Two actual full-tensor witnesses show what is and is not established.

1. Set delta P=Omega*q, other scalar/gauge/metric deviations zero, and

```
l=4q/(3kappa*a^2+2), A_r=-2l/3,
delta Lambda=l*n,
delta A=A_r*(1.5n n-.5I).
```

All20 R0 residues vanish, Q is finite and Nraw remains the reference's quadratic-null value initially. Nonetheless

```
Theta_t0=-2q/a^2,
Omega*Q_t -> (2q-3l)/a^2,
Nraw_t0=2(l-2q)/(3a^3).
```

This witness violates asymptotic constraints (M/Z boundary residues are recorded), so it is not a vacuum Einstein counterexample. It shows finite Q/null values and complete pole cancellation do not alone give smooth compatible time derivatives.

2. Set only delta P=Omega^2*q. Its initial R0 and all scalar boundary rates above vanish, with H/M/Z/Theta boundary values zero. Its exact first actual RHS is

```
alpha_t=-alpha_hat^2 Omega*q,
chi_t=2alpha_hat Omega*q/3,
P_t=-2Omega^2*q/a,
Theta_t=-2alpha_hat Omega*q/a,
Lambda_t^i=8alpha_hat*x^i*q/(3a),
beta_t=g_t=A_t=0.
```

Use that exact Cartesian field, including its derivatives, as the next dual seed. The leading pole derivative is nonzero:

```
(d_t R0)_A,nn = -8q/(3a^4),
(d_t R0)_Lambda,n = (12-8kappa*a^2)q/(3a^4).
```

Thus the initial R0 and scalar first-corner conditions do not close the full jet compatibility hierarchy. The next necessary condition is

```
d_t R0 = B(F0,F1,tangential F0)=0,
```

where F0/F1 are the first Taylor coefficients of the actual assembled RHS. They involve second spatial Taylor jets. Further preservation may require additional hierarchy or a subsidiary-compatible boundary construction; none is derived here. This is not permission for a value-only ghost projection onto five zero modes.

Failure of smooth corner compatibility does not by itself prove finite-Q amplitude blowup. There is a precise actual-matrix reason for this distinction. For the reference value-only pole matrix P, let char(P)=lambda^5 Q(lambda). Its reconstructed exact kernel projector is Pi=Q(P)/Q(0), and D=(P+Pi)^(-1)-Pi is its group inverse. The compiled/rational check proves Pi^2=Pi, P*Pi=0, P*D=I-Pi, and the Theta and delta(P-3w) rows annihilate Pi. The same actual matrices have Hurwitz nonzero roots in the separately frozen local pole gate.

In the frozen normal model u_t=P*u/Omega+f with finite constant forcing and zero initial data,

```
u(t)=t*Pi*f + Omega*D*(exp(tP/Omega)-I)*(I-Pi)*f.
```

Consequently Theta and delta(P-3w) can have O(1) initial time derivatives while their amplitudes stay O(Omega), through an initial time layer t~Omega. This statement uses the actual20 pole matrix, not a substitute scalar equation. It is deliberately not a PDE estimate: the real pole operator also contains first spatial derivatives, time-dependent Taylor jets and variable coefficients; bounded forcing/semigroup control for the complete spatial problem has not been established. Neither amplitude blowup nor smooth scri closure follows from the two witnesses alone.
