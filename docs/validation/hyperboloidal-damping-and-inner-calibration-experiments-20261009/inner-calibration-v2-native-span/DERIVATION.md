# Inner Schwarzschild trumpet calibration for the actual core gauge

This is a separate analytical preparation for a later BH goal. The Minkowski reference remains unchanged; there is no BH RHS subtraction, native BH evolution or proposed production edit. The exact inner geometric plateau has Omega=1, reference alpha=chi=1, beta=K=Lambda=0 and zero reference derivatives. With lapse_inner=0 and W_gauge=0, the actual physical-P lapse equation on the Einstein sector P=K,Theta=Z=0 reduces to

```
alpha_t = beta^i alpha_i - alpha(alpha+2) K.
beta_t^i = beta^j partial_j beta^i
           +alpha^2 chi (3q0/4) Lambda^i - eta_inner beta^i.
```

Both lapse flags give the same core equation. These statements require the exact inner plateau, not the nonflat geometric transition.

## Signs and stationary areal solution

Use Kij=-(1/2)L_n gammaij, our numerical-relativity sign, and the future black-hole branch with outward coordinate shift. Starting from Schwarzschild F=1-2M/R and a stationary height transformation T=t+h(R), set

```
q=sqrt(alpha^2-F)>0,
gamma_RR=alpha^-2, beta^R=alpha*q,
h_R=-q/(alpha*F),
K^R_R=q_R, K^theta_theta=K^phi_phi=q/R,
K=q_R+2q/R=(R^2 q)_R/R^2.
```

The height derivative has a Schwarzschild-coordinate horizon singularity; lapse/shift/spatial ADM quantities in this stationary gauge remain finite there. Stationary BM slicing yields q alpha_R=(alpha+2)K, hence

```
q=C(alpha+2)/R^2,
alpha^2=1-2M/R+C^2(alpha+2)^2/R^4.
```

For this exact family the ADM Hamiltonian and momentum constraints vanish identically. The proof independently checks R3=2(1-alpha^2-2R alpha alpha_R)/R^2 and the radial K eigenvalues above.

Let t=M/Rc. Smooth critical crossing requires both numerator and denominator of the differentiated implicit relation to vanish:

```
alpha_R=[M/R^2-2q^2/R]/[alpha-q^2/(alpha+2)].
q_c^2=t/2=alpha_c(alpha_c+2), alpha_c=t-1/2,
4t^2+2t-3=0.
t=(sqrt13-1)/4,
Rc/M=(sqrt13+1)/3,
alpha_c=(sqrt13-3)/4,
C^2/M^4=8(13sqrt13-35)/243.
```

The positive critical slope is selected from
`4z^2/(alpha_c+2)+16alpha_c z-6t=0`, z=Rc alpha_Rc. The other sign does not give the selected increasing branch.

For M=1, define Q3(R)=R^3+(2Rc-2)R^2+(3Rc^2-4Rc)R+4Rc^3-6Rc^2. All coefficients are positive, and the discriminant factors as R^3(R-Rc)^2 Q3. The smooth physical root is

```
alpha(R)=[2C^2+(R-Rc)sqrt(R^3 Q3(R))]/(R^4-C^2), R>=R0.
```

Using the *signed* factor R-Rc avoids an artificial cusp from sqrt((R-Rc)^2). This explicitly switches quadratic roots at the critical point. The zero-lapse equation is the **quartic** R0^4-2MR0^3+4C^2=0. Its smaller positive root below Rc is selected by alpha_R0>0; the larger root has negative zero-lapse slope and is not the endpoint of this branch.

These BM implicit/critical formulas agree with Eqs.28–31 of [Ohme et al., arXiv:0905.0450v2](https://arxiv.org/pdf/0905.0450), after reversing that paper's extrinsic-curvature sign. Its nearby printed R0≈1.3955M is inconsistent with those equations; the reproduced quartic gives 1.3195497562M and this is the equation-based target.

## Isotropic endpoint and required driver rate

For an isotropic Cartesian spatial metric gammaij=(R/r)^2 deltaij,

```
dr/r=dR/(alpha R), chi=(r/R)^2, gtildeij=deltaij, Lambda^i=0,
beta^r=r*q/R.
```

At R0,

```
q0=sqrt(2M/R0-1)=2C/R0^2,
alpha_R0=2(3M-2R0)/(q0^2 R0^2)>0,
p=R0 alpha_R0,
v=q0/R0,
K0=(3M-2R0)/(q0 R0^2), p*v=2K0.
```

The exact positive slope gives R-R0~B r^p, alpha~alpha_R0 B r^p, beta^r~v r, chi~r^2/R0^2. Proper radial length diverges logarithmically while areal radius approaches nonzero R0: a trumpet. The free isotropic radial normalization changes B, not p,v,K0. These differentiated asymptotics belong to the selected exact smooth areal branch; they are not an imposed falloff condition on arbitrary evolved data.

With Lambda=0, the current first-order driver is stationary only if

```
eta_required(R)=partial_r beta^r
 =(q/R)[1-3alpha+alpha R alpha_R/(alpha+2)].
```

Its endpoint limit is v, so eta_inner=v is necessary for cancellation of the leading O(r) residual. It is **not sufficient** for a fully stationary isotropic metric:

```
eta_required_R(R0)
 =v[(R0 alpha_R0^2-5alpha_R0)/2-3/R0]
 =-2.09890361831674562/M^2 != 0.
```

With constant eta=v, the nonzero next shift RHS is beta^r[eta_required(R)-v]=O(r^(1+p)). A stationary solution in different radial coordinates could have nonflat conformal metric and nonzero contracted connection; this calculation neither excludes nor constructs that solution. Setting an independent nonzero Lambda on the same isotropic metric would introduce a Z constraint and does not solve the Einstein-sector gauge problem.

## Numerical constants and implementation limitations

```
R0/M = 1.3195497562204374803391352194
p = 1.0607696620240810291490520736
M*v = 0.5442012447587540694205747556
M*K0 = 0.2886360852379138758991867991
eta_inner(M=.5) = 1.0884024895175081388411495113
```

The actual current validator requires shift_outer>=shift_inner; default shift_outer=1 therefore cannot accept this M=.5 leading calibration without also increasing that rate (or separately revisiting the monotonic restriction). No such change is made here. For the wide geometric plateau r<=.05 and native span2.1 cell-centred Cartesian grid, the minimum radius is sqrt3*2.1/(2N): N24/36 have no exact-core cells and N48 has only the central eight. Existing Minkowski pulse runs do not resolve this asymptotic calibration problem.

The symbolic/80-digit audit and 30 pointwise actual `InteriorLayerGauge` checks preserve the exact Minkowski core reference. Numerical radial normalization r(R=2M)=.01 is arbitrary and used only to place check points in that exact core. These are gauge function evaluations, not a spherical replacement for native evolution. There is no claim of wormhole-to-trumpet formation, global hyperboloidal/BH matching, nonlinear attraction, or uniform puncture hyperbolicity; alpha and chi degenerate at the limiting puncture and that problem remains separate.
