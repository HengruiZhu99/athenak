# Constraint coefficients and independent gauge/null corner conditions

This is a linear actual-kernel audit on the outer pure-CMC Minkowski reference, S=1, a=.5,.75,1,2, kappa_input10, physical-P lapse and spatial-norm shift (xi=1/a,rho1.5). Both kappa2=0 and the live coefficient's outer reference value are tested. It neither closes the nonlinear hierarchy nor classifies the native pulse growth.

Use Cartesian components, n=x/r, Omega=(1−r²)/(2a), alpha_ref(r)=(1+r²)/(2a), beta_ref=−x/a, gtilde_ref=I, chi_ref=1, A_ref=0 and P_ref=−3/a. In this note alpha_ref(r) always denotes the full spatial function; its boundary value is1/a. Write each perturbation f=f0(n)+Omega*f1(n)+Omega²*f2(n)+..., so f1/f2 are Taylor coefficients. Put h=delta gtilde (tracefree), c=delta chi, K=delta P+2delta Theta, X=c−h_nn. Directional sphere derivatives below act on Cartesian components: D_i=(delta_i^j−n_i*n^j)partial/partial n^j. Normal components such as h_nn are contractions, whose differentiated n factors must be retained.

The exact linear physical constraint formulas, independently matched to4000 actual dual EvolvedConstraints evaluations, are

```
Rlin = partial_i partial_j h_ij + 2 Delta c,
L = delta(Delta_bar Omega) = [−3c+x_k(partial_j h_jk+partial_k c/2)]/a,
Glin = r² X/a²,
H = Omega² Rlin −4K/a +4Omega L −6Glin,
M_i = Omega partial_j A_ji −(2/3)partial_i K +(2r/a)A_in,
Z_i = [Lambda_i−partial_j h_ji]/2,
Theta_phys = theta.
```

Here H is the physical Hamiltonian and M/Z are the native covectors. The reference conformal norm of M is not its physical norm: the latter has an additional Omega² factor. Vanishing coefficients below express Einstein-compatible jets, not a condition inferred from bounded physical norms of arbitrary constraint violations.

For L=L0+Omega L1+... and X=X0+Omega X1+Omega²X2+..., the Hamiltonian coefficients are

```
H0=−4K0/a−6X0/a²,
H1=−4K1/a+4L0−6X1/a²+12X0/a,
H2=Rlin0−4K2/a+4L1−6X2/a²+12X1/a.
```

Smooth R0-compatible boundary values from the previous actual pole audit obey alpha0=P0=Theta0=beta0=0 and X0=0; their tangential derivatives also vanish. Thus H0=0 automatically. On this subspace,

```
M_i0=(2K1/(3a))n_i+(2/a)A_i n,0,
M_i1=A_i n,1/a+D_j A_ji,0−2A_i n,0
      +(4K2/(3a)−2K1/3)n_i−(2/3)D_i K1,
Z_i0=[Lambda_i0+h_i n,1/a−D_j h_ji,0]/2,
Z_i1=[Lambda_i1−h_i n,1+2h_i n,2/a−aD_j h_ji,0−D_j h_ji,1]/2.
```

Theta's coefficients are its stored coefficients. The complete18×200 maps in check-report.json give H0/H1/H2, M0/M1, Z0/Z1 and Theta0/Theta1/Theta2 on the ten local Taylor/angular monomials (1,Omega,n_y,n_z,Omega²,Omega*n_y,Omega*n_z,n_y²,n_y*n_z,n_z²), with20 independent fields per monomial. This spans the full second spatial jet at n=(1,0,0), including the nonfree derivatives of n and Omega. Quadratic angular monomial coefficients are half the diagonal angular second derivatives. Each of the four exact rational-a maps has rank18. M2/Z2 are not supplied: they would require higher jets. General formulas are analytic; the exact map rank statements concern the four sampled rational radii.

The pole-Lambda equation gives an exact useful identity before imposing any constraint falloff:

```
M_i0−(a*kappa_input−2/a)Z_i0+partial_i Theta|0
    =(a/2) R0_Lambda_i.
```

For a smooth R0-compatible field Theta0(n)=0, partial_i Theta|0=−Theta1*n_i/a. Therefore

```
M_i0=(Theta1/a)n_i+(a*kappa_input−2/a)Z_i0.
```

Consequently Einstein-compatible M0=Z0=0 implies Theta1=0. This is a derived compatibility consequence; no stronger Theta falloff is imposed on the arbitrary off-constraint system.

The earlier corrected actual scalar corner formula is

```
Theta_t0=(2/a)L0−(2/a²)delta Q−kappa_input(2+kappa2*)Theta1−3Nraw1/a,
delta Q=P1−3(alpha1+beta_n1),
Nraw1=X1/a²+2(alpha1+beta_n1)/a.
```

Substitution of H1 cancels the gauge first-jet combination exactly:

```
Theta_t0=H1/(2a)+[4/a²−kappa_input(2+kappa2*)]Theta1.
```

Thus H1=M0=Z0=0 supplies this Theta corner rate. The exact symbolic identity and160 actual R0-compatible first-jet tests agree (M identity1.11e−16; scalar corner2.78e−14). kappa2*=0 for baseline and2/(kappa_input*a²)−1 for the live outer reference. No variable-coefficient or finite-amplitude constraint energy follows from these identities.

## Why the Omega² P witness is off the next Einstein jet conditions

With only delta P=Omega²*q,

```
H=−4q Omega²/a,
M_i=4q r Omega n_i/(3a), Z=Theta=0.
```

H0=H1=M0=Z0=Theta0=Theta1=0, but H2=−4q/a and M1=4q*n/(3a) are nonzero. The previous actual next-RHS pole obstruction therefore occurs after nonzero higher Einstein-constraint coefficients. This audit identifies those coefficients; it does not prove that setting H2=M1=Z1=Theta2=0 is sufficient for all next pole or gauge compatibility conditions.

## An independent initial gauge/null obstruction even on Einstein initial data

Set only delta alpha=Omega and delta beta=−Omega*n. The spatial metric, extrinsic-curvature variables and P/Theta remain exactly their reference initial data. Hence H/M/Z/Theta vanish identically as functions of radius, not merely at the boundary, and every initial pole residue vanishes. At linear order the actual stationary-Omega normal gives

```
delta Nraw=−a Omega³+O(Omega⁴),
delta Q=(3a²/2)Omega²+O(Omega³).
```

The initial quadratic-null conditions Nraw0=Nraw1=0 and finite Q hold (even Nraw2 and Q0/Q1 variations vanish). Nevertheless the actual full20 RHS limit is

```
alpha_t0=−3/a², beta_n,t0=3/(2a²), P_t0=3/a², Theta_t0=0,
chi_t0=−2/(3a), (h_nn)_t0=4/(3a),
Nraw_t0=−5/a³, (P−3w)_t0=15/(2a²).
```

The independently extrapolated actual limit agrees with these scalar formulas to7.11e−14 over four a and both damping forms. The scalar pole itself then has (d_t R0)_alpha=15/(2a⁴), already nonzero without requiring a differentiated RHS spatial jet. Thus zero Einstein constraint coefficients alone do not preserve the smooth fixed-reference gauge/null corner. A subsequent boundary continuation must enforce independent gauge/null time-jet compatibility too.

This is an initial-data, linear time-tangency statement. It is not a vacuum spacetime singularity, a proof of finite-Q amplitude blowup or a diagnosis of the current compact native pulse. Stable fast relaxation may create a temporal boundary layer; controlling that mechanism still requires the full spatial hierarchy and an appropriate energy/boundary argument. No ghost rule, production falloff, Omega floor or new runtime option follows from this audit.
