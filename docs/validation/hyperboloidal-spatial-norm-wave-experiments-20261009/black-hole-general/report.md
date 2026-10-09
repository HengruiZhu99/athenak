# General outer shift-second-jet compatibility for detached BH data

Scratch-only result, 2026-10-09. No production source changes or native BH
evolution. The preceding detached geometry/beta-only 69-file archive and
spatial-norm 34-file gate are unchanged and verified by this audit's runner.
The objective is a reusable necessary condition for eventual BH initialization,
after the finite Minkowski pulse gate; no preserved scri manifold is proved.

## General correction

Retain the actual Minkowski reference and its monotone compactification.
Give the BH height its own cutoff with its throat inside the exact Cauchy
height core. Both cutoffs reach their exact outer CMC branches before scri.
The derived outer BH fields are

```text
Omega=(S^2-r^2)/(2aS),    L=S/a-Omega,    m=M*Omega/(2r),
psi=1+m,  alpha_geo=L*(1-m)/psi,
beta_geo_rad=-(r/a)*(1-m)/psi^3,
chi=psi^(-4),  gtilde=I,  Lambda=0,
P=-(3-2m+m^2)/[a*(1-m)*psi^3].
```

The initial metric, physical trace/shear and Cartesian connection remain
the BH data's own fields. All gauge reference values remain Minkowski.
Use preferred source off and

```text
xi=1/a,  eta=rho*S/a^2,  C=(S/a)*(1-1/rho),
S_beta^i=-eta W[beta^i-beta_ref^i+C*n^i*(G/Ghat-1)],
G=chi*gtilde_inverse^ij*Omega_i*Omega_j.
```

Keep the production harmonic outer lapse/shift principal terms. Let
nu=lapse_outer and eta_regular=shift_outer denote their existing regular
restoring rates. Production defaults are nu=1.5 and eta_regular=1 in code
length units; they cannot be omitted from a dimensionally general result.

Add a radial second-order gauge tail

```text
beta_initial^i = beta_geo^i + tau(r)*D*Omega^2*n^i,
tau=SmoothCutoff(r,.97*S,.99*S).
```

The necessary unique coefficient when rho!=4 is

```text
D = M*[M*(rho-8)+4*a^2*(2*eta_regular-nu)+8*a*(4-rho)]
      / [4*S*a*(4-rho)].
```

This yields D=29/40 at S=1,a=M=.5,rho=1.5 with default regular rates.
At rho=1 it reduces to the earlier beta-only balanced-rate correction
D=3/4 for those same S,a,M and regular rates. Any smooth tau that is one
in a neighborhood of scri has the same boundary jet result. Its support
must avoid the puncture; the audit uses the exact outer neighborhood above.

`general_beta2.hpp` exposes the scratch host API
`outer_beta2::Coefficient(S,a,M,rho,nu=1.5,eta_regular=1)`. It validates
finite parameters, S>0,a>=S/2,M>=0,rho>0 and nonnegative regular rates;
it throws at rho=4 because a unique formula is unavailable there. The API
does not validate the separate height/throat condition or assert that every
positive rho has the admitted nonlinear gauge branch or a stable symbol.
The proposed branch range remains 1<=rho<=5/2. No parameter floor or clipping
is used.

## Derivation and limits

Reserve Q=(P-3*omega_n)/Omega for the conformal trace. The distinct null
quantity is N_raw=G-omega_n^2, with omega_n=-beta^i*Omega_i/alpha.
Its complete time derivative retains both the chi/metric and gauge terms.
The original first jets alpha-alpha_ref=-(M/a)*Omega+... and
beta_rad-beta_ref_rad=(2M/a)*Omega+... give all leading gauge/null/geometric
rates zero for xi=1/a and the stated eta/C relation, at general S,a,M,rho.
The beta second-jet correction does not change those first jets.

Independent exact series of the full geometric/gauge projection give

```text
dt N_raw = [B0 + 2*(4-rho)*D/a^3]*Omega + O(Omega^2),
B0 = -M*[M*(rho-8)+4*a^2*(2*eta_regular-nu)+8*a*(4-rho)]
       / [2*S*a^4].
```

An additional lapse coefficient delta_a2*Omega^2 contributes zero to this
Omega coefficient. This audit keeps delta_a2=0 and retains the positive
initialized lapse. The formula for D follows by cancelling the displayed
coefficient, rather than setting the complete geometric contribution to zero.
With the corrected initial shift,

```text
N_raw = [1/S^2+2D/(aS)]*Omega^2+...,
Q_scri = -3/S+M/(aS),
dt N_raw = O(Omega^2).
```

S,a,M have length units, nu/eta_regular/xi/eta have inverse length units,
and D is dimensionless. The bracket defining D has length units, so its
numerator and denominator both have length squared units. The B0 coefficient
has inverse length cubed units. Exact symbolic tests verify that
S,a,M -> ell*(S,a,M), nu/eta_regular -> (nu/eta_regular)/ell leaves D
invariant and scales B0 by ell^(-3). Keeping numerical regular rates fixed
while rescaling geometry changes the physical gauge and therefore changes D.

The separate BH geometric admissibility condition is M>0 and
M<2*r_height0/Omega(r_height0), so physical R=M/2 lies strictly inside its
exact Cauchy height core. The layer compactification also requires a>=S/2
and 0<r_layer0<r_layer1<S. A future native puncture mesh must still exclude
the exact origin. The correction does not change any ADM field or the mass.
It changes the coordinate shift from the static outer Schwarzschild gauge;
the resulting data are not a stationary evolution.

For rho!=4, D smoothly tends to zero as M tends to zero. At positive M,
D is zero precisely when its numerator bracket vanishes; that is already
an initial next-null-jet compatibility point, not a stationarity proof.
The moderate tested masses have D>0. For more general admissible height
cores D can have either sign. If positive spatial gradient norm N_raw is
also desired just inside scri, the extra necessary leading condition is
1/S^2+2D/(aS)>0. Equality needs the next coefficient; negative values do
not give that causal character. A large finite D may require a narrower
gauge-tail collar to retain the desired shift orientation away from scri.
These are coordinate-gauge choices, not changes to the vacuum ADM constraints.

At rho=4 the second-jet response vanishes, and

```text
dt N_raw = [2M*(M-a^2*(2*eta_regular-nu))/(S*a^4)]*Omega+... .
```

For positive M unequal to a^2*(2*eta_regular-nu), neither a radial beta
Omega^2 correction nor a lapse Omega^2 correction can repair this coefficient
with the fixed first/ADM jets. This is a genuine second-jet obstruction in
that restricted family. If M=a^2*(2*eta_regular-nu)>0, the coefficient
already vanishes and D is nonunique. For default rates at S1,a.5,
M=.5 gives the uncancellable +6*Omega term; M=.125 gives the degenerate
nonunique case. The 100-digit oracle verifies both. The proposed
1<=rho<=5/2 interval stays away from this degeneracy.

## Initial time compatibility of the exposed poles

The frozen `../spatialnorm-gate/norm_leading.py` proof is already symbolic
in S,a,M,eta and D; its bytes are unchanged. The general D changes no
leading alpha/beta/chi/g/P/Theta rates. Both gauge pole time derivatives
therefore remain zero at the initial instant. The complete new geometric
time jets are

```text
Lambda_dot^r = Gamma_tilde_dot^r = 8D/(3a^2),
partial_r chi_dot  =  2D/(3a^2),
partial_r g_rr_dot =  8D/(3a^2),
partial_r g_tt_dot = -4D/(3a^2).
```

The physical-spatial connection time derivatives are both D/a^2 in the
radial and tangential slots, making Hess(Omega)'s time change isotropic at
scri. The mass-dependent finite conformal A is retained. P advection changes
by D*Omega^2*P_r, A Lie/advection changes by O(Omega), and Theta remains
zero for the exact initial vacuum state. Hence P_dot0,P_dot_r0,
Theta_dot0,Theta_dot_r0,A_dot0 and Z_dot0 are zero. The initial time
derivatives of the chi/P/Theta/A/Lambda pole numerators vanish, just as in
the earlier general-symbol proof. Lambda must not be pinned to zero: its
nonzero rate follows the evolving contracted metric connection.

The actual-kernel general audit independently checks the Lambda Hessian
increment and the metric/chi first time jets for all eight cases. No new
first/initial-time-pole obstruction appears in the tested admitted rho range.
The statements concern one instant; the hierarchy at later times and under
general angular/off-constraint perturbations is unproved.

## Independent checks

`general_second.py` derives the general coefficients symbolically from the
outer fields and full projection, checks dimensions and the rho4 obstruction,
and reduces D to29/40. `general_oracle.py` independently differentiates the
factored full rate at 100 digits, solves its beta linear response numerically,
and compares that solution with the closed formula. The oracle keeps the
original/corrected rate rows and the negative rho4 controls.

| S | a | M | rho | nu | eta_regular | D | Corrected dt N_raw / Omega^2 limit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | .5 | .5 | 1.5 | 1.5 | 1 | .725 | -92.7 |
| 1 | .5 | .2 | 1 | 1.5 | 1 | .37 | -34.584 |
| 1 | .75 | .35 | 1.5 | 1.5 | 1 | .6463333333 | -27.26584691 |
| 1 | 1 | .5 | 2.5 | 1.5 | 1 | .9375 | -21.5 |
| 2 | 1 | 1 | 1.5 | 1.5 | 1 | .775 | -12.9875 |
| 2 | 2 | .3 | 2 | 1.5 | 1 | .358125 | -.971671875 |
| .8 | .8 | .16 | 1.25 | 1.5 | 1 | .4045454545 | -16.92879972 |
| 2 | 1 | 1 | 1.5 | .75 | .5 | .725 | -11.5875 |

The last row is the first row rescaled by two in length. D is unchanged
and its rate coefficient is divided by eight, as required. Every actual
kernel case uses the scaled .05*S–.95*S Minkowski reference, .30*S–.95*S
detached BH height, .45*S–.85*S gauge cutoff and preferred source off. The
outer shape is unchanged; only the stated geometric/rate parameters vary.

`kernel_general.cpp` consumes the actual tensor and gauge kernels and the
reusable `general_beta2.hpp` API. Release and Debug ASan/UBSan both pass,
with eight configurations and 112 rate rows per executable plus oblique
consumed-jet and independent time-connection checks. Maximum H/M residuals
are 1.954e-14/2.220e-15, exact scri pole residuals 4.026e-15, and direct
raw/factored rate agreement 9.399e-11 where Omega>=1e-4. Consumed beta
first/second derivative relative errors are 3.959e-10/4.515e-4 at FD step
1e-6*S; the independent connection/time-jet error is 6.106e-9.

`receipt.json` records four passing checks (two actual-kernel modes and
two independent proof/oracle scripts), 3.16017 seconds of test execution,
5.51033 seconds including builds, source HEAD
`f615acf4356206eceddc59fa929fcc15a671fe09`, exact compiler commands and hashes.
Release SHA256: `f1aabf309172224d7a174d1e415cd4e52856c8c69309ff3df1bb6776000d6a07`.
Debug SHA256: `b5fd538eede17943db464d8e76848f546d6f5feca9e04c8dfd3ecb437726a386`.
The Debug run keeps Address and UndefinedBehavior sanitizers enabled;
macOS-unsupported LeakSanitizer is disabled explicitly.

The exact-scri checks inspect pole numerators and factored limits, without
assembling pole/Omega at Omega=0. Tiny-Omega raw rows are retained, including
cancellation amplification; they do not establish limiting accuracy. The
100-digit oracle supplies the quadratic coefficients independently. No
native BH simulation, BH RHS subtraction, automatic gauge coupling,
production edit or dynamic boundary-closure claim is part of this result.
