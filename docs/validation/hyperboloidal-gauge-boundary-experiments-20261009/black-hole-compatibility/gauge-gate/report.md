# Later-BH outer gauge gate: instantaneous null-rate audit

The eta=10 shift pole fails the later-BH boundary compatibility gate for the fixed detached M=.5 ADM data with the original quadratic null falloff. At S=1, a=.5, physical-P lapse xi=1.5 and the unchanged wide Minkowski reference (.05,.95), the full initial null-rate limit is +24 for the source-off baseline and -56 after the proposed shift pole. Gauge first-jet edits can cancel that rate, but immediately violate the added pole's required shift boundary value. The rederived preferred projection/null feedback cancel the null rate but remain a separate previously rejected source family. This analysis neither changes nor settles the native finite-pulse Minkowski experiment.

All results here are scratch only. The BH ADM initial data use the independently audited detached height (.30,.95), with their own gtilde/Lambda. The actual Minkowski geometry/gauge reference is unchanged. No BH reference RHS is subtracted.

## Quantities and coordinate convention

Reserve the production trace variable

```
Q = (P-3*omega_n)/Omega
```

and call the distinct null quantity

```
N_raw = chi*gtilde^{ij}*Omega_i*Omega_j - omega_n^2,
omega_n = -beta^i*Omega_i/alpha.
```

Omega is time independent, alpha is the conformal lapse, and beta is the contravariant coordinate shift. A radial shift means `beta^i=beta_r*n^i`, with n the Euclidean Cartesian radial unit vector; beta_r is the compact-coordinate component, not an areal-radius component. At outer scri, `Omega'=-1/a`, `alpha0=S/a`, `beta_r0=-S/a`, `omega_n0=-1/a`. The initial BH Theta and Z are zero, so kappa does not affect this instantaneous rate.

For `v^i=gtilde^{ij} Omega_j`, differentiate every geometric and gauge term:

```
dt N_raw = (dt chi)*gtilde^{ij} Omega_i Omega_j
           -chi*v^i*(dt gtilde_ij)*v^j
           +2*omega_n/alpha*(dt beta^i*Omega_i+omega_n*dt alpha).
```

The first two terms use the actual unsubtracted conformal geometric equations. The last terms use the full physical-P lapse and coupled shift, including every shift pole. For the original stationary exterior lapse/shift, the geometric contribution is zero; it is generally nonzero for altered initial gauge jets and must be retained.

## Factoring the initial BH tail

In the outer collar let `L=(S^2+r^2)/(2*a*S)`, `b=r/a`, `m=M*Omega/(2r)`, `psi=1+m`, `N=(1-m)/psi`, `F=(1-m)/psi^3`. The geometric outer data are `alpha=N*L`, `beta_r=-b*F`, `chi=psi^-4`, `gtilde=I`. Rather than subtract nearly equal values and divide by Omega, use

```
(alpha-alpha_ref)/Omega = -M*L/(r*psi),
(beta_r-beta_ref_r)/Omega = M/(2*a)*(4+3m+m^2)/psi^3,
(chi-1)/Omega = -M/(2r)*(4+6m+4m^2+m^3)/psi^4,
(P-P_ref)/Omega = M/(2*a*r)*(8-m-6m^2-3m^3)/[(1-m)*psi^3].
```

For free gauge tails `da_div`, `db_div`, the normal difference also factors exactly:

```
(omega_n-omega_n_ref)/Omega
   = -(db_div*Omega'+omega_n_ref*da_div)/alpha,
Q = -3/(a*L)+(P-P_ref)/Omega
    -3*(omega_n-omega_n_ref)/Omega.
```

Thus neither the normal trace Q nor its limit is confused with N_raw. The original data have `N_raw=Omega^2*psi^-4*(Omega')^2/L^2` and `Q_scri=-3/S+M/(a*S)=-2` for the audited parameters. The shift deviation being O(Omega) is proved for this initial profile; it is not assumed as a falloff or closure for arbitrary evolved fields.

The proposed additional numerator is `S_beta^i=-eta*W*(beta^i-beta_ref^i)`, with the unchanged production gauge weight W (.45,.85), outer regular shift damping 1, lapse restoration 1.5, xi=1.5 and eta=10. It has a finite, nonzero initial limit after division by Omega. For the original BH jets it changes beta_dot_scri from +8 to -12 while alpha_dot_scri stays -2.

| Gauge with original geometric BH outer jets | alpha_dot at scri | beta_r_dot at scri | dt N_raw at scri |
|---|---:|---:|---:|
| Physical-P, source off | -2 | +8 | +24 |
| Rederived preferred projection, sigma=0 | -2 | +2 | 0 |
| Same projection plus null feedback, sigma=5 | -2 | +2 | 0 |
| Source off plus eta=10 shift pole | -2 | -12 | -56 |

The 100-digit outer calculation additionally gives, at these specific parameters,

```
baseline:  dt N_raw = 24 -54*Omega +9*Omega^2 +...,
preferred: dt N_raw = -16*Omega^2 +...,
feedback:  dt N_raw =  24*Omega^2 +...,
eta10:     dt N_raw = -56 +31*Omega +14*Omega^2 +....
```

The preferred/null-feedback rows refer to the archived, rederived physical-P source, not the prohibited reuse of the original production preferred source. Their initial BH compatibility does not reverse the existing negative Minkowski frozen-pole/native evidence for that source family.

## Free first jets and the trace Q condition

Write outer gauge deviations as

```
alpha-alpha_ref = a1*Omega+O(Omega^2),
beta_r-beta_ref_r = b1*Omega+O(Omega^2).
```

The fixed BH ADM data have `chi=1-(2M/S)*Omega+...`, `P=-3/a+(4M/(a*S))*Omega+...`, `A_rad=-4M/(3*a*S)` at scri. The original geometric gauge gives `a1=-M/a`, `b1=2M/a`. An independent symbolic projection of the full equations gives

```
N_raw = n1*Omega+O(Omega^2),
n1 = 2/(a*S)*(a1+b1-M/a),

dt N_raw|scri = 2/a^3*[(2-eta*a^2/S)*b1
                      -2*xi*a*a1-4M/a].
```

These formulas include the geometric term, which is `-2/a^3*(a1+M/a)` at scri. For example the compatible eta10 jets below have a geometric null-rate contribution +56, canceled by gauge contribution -56; checking alpha_dot+beta_dot alone would give the wrong conclusion.

For preserving this initial slice's stronger mass-corrected quadratic null falloff, impose `n1=0`, hence `a1+b1=M/a`. The trace Q is finite already from `P0=3*omega_n0`. Its finite first-jet limit is separately

```
Q_scri = -3/S+4M/(a*S)-3*(a1+b1)/S.
```

The condition `a1+b1=M/a` therefore keeps the original trace limit `-3/S+M/(a*S)` as well as `N_raw=O(Omega^2)`. These are related initial jet conditions on two distinct quantities, not a claim that Q is the null residual.

Solving both `n1=0` and `dt N_raw|scri=0` for the source-off eta family yields

```
a1 = -(2+eta*a^2/S)/(2+2*xi*a-eta*a^2/S) * M/a,
b1 =  (4+2*xi*a)/(2+2*xi*a-eta*a^2/S) * M/a.
```

Dimensions are consistent: S,a,M have dimensions of length; Omega, alpha, beta_r, a1 and b1 are dimensionless; xi and eta are rates with dimensions inverse length. Therefore xi*a and eta*a^2/S are dimensionless, Q has dimension inverse length, N_raw inverse length squared, and its time derivative inverse length cubed. At the audited numbers, eta10 gives `a1=-4.5`, `b1=5.5`, instead of the original -1,+2. The denominator vanishes at eta=14 for these parameters; no such pair exists there for positive M. Near that value the compatible jets become large.

For the unchanged geometric first jets, the first tangency equation alone would require `eta=xi*S/a=3`, not 10. That single equality still does not establish preservation of the quadratic null falloff or a nonlinear boundary manifold.

The rederived preferred source instead gives `dt N_raw|scri=(4*S/a^2)*n1`; its sigma feedback changes this to `(4-2*sigma)*S/a^2*n1`. Both vanish on `n1=0` at this instant. They do not constitute a stability proof or justification for adopting that rejected source family.

## A partial null-only construction at one instant

Changing only the initial lapse/shift can satisfy the first gate without changing the physical spatial metric, K_ij, ADM mass or initial hypersurface. Use a smooth radial collar tau, zero through .97 and one after .99:

```
alpha_new = alpha_geometric + tau*(-3.5*Omega + delta_a2*Omega^2),
beta_r_new = beta_r_geometric + tau*(3.5*Omega + delta_b2*Omega^2).
```

Their trace Q still has the original finite limit -2, and their initial N_raw is O(Omega^2). This is only a null-rate exercise: the construction fails the shift-pole boundary-value gate below. Exact rational second-jet algebra with the current gauge rates gives

```
dt N_raw = [-575/2 +8*delta_a2+24*delta_b2]*Omega +O(Omega^2).
```

Setting both second changes to zero cancels only the constant null rate and leaves `dt N_raw=-287.5*Omega+...`. One choice `delta_a2=575/16=35.9375`, `delta_b2=0` cancels that next coefficient; the 100-digit calculation then gives `dt N_raw=-6406*Omega^2+...`. These are compatibility conditions on the initial jet at one instant. The constructed scri lapse/shift rates are +33 and -47 and the metric is initially dynamical. Large rates and steep collar derivatives make this an existence example, not a recommended BH initial gauge or an evolution result.

No preservation theorem follows: subsequent time derivatives involve the evolved metric, P, A, Theta/Z and additional spatial/gauge jets. A finite pole limit, a frozen stable matrix or satisfaction of these two initial conditions does not prove that the evolved fields retain them. The original height/mass geometry remains valid as an ADM slice, while the changed lapse/shift no longer represent its stationary Schwarzschild time vector.

## Boundary-value obstruction from finite gauge poles

For eta>0 the additional pole numerator itself requires `beta_scri=beta_ref` if the smooth finite RHS is to persist. Its tangent condition is therefore `beta_dot_scri=0`; merely arranging dt N_raw=0 does not suffice. The full coupled shift yields

```
beta_dot_scri=S/a^2*(a1+b1+M/a)-eta*b1.
```

With `a1+b1=M/a`, eta10 pinning demands a1=.2,b1=.8. Its full null rate is -75.2, whereas null tangency demands a1=-4.5,b1=5.5 and gives beta_dot=-47. This is an overdetermined first-jet obstruction for this fixed mass data/quadratic null falloff. No second-jet change can repair a first-jet contradiction.

The actual geometric leading rates were retained, not pinned artificially:

| eta10, xi1.5, initial first jets | alpha_dot | beta_dot | chi_dot | gtilde_rad_dot | gtilde_tan_dot | P_dot | dt N_raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original -1,+2 | -2 | -12 | 0 | 0 | 0 | 0 | -56 |
| Null-only -4.5,+5.5 | +33 | -47 | 14/3 | -28/3 | 14/3 | -42 | 0 |
| Shift-pin .2,+.8 | -14 | 0 | -8/5 | 16/5 | -8/5 | 72/5 | -376/5 |

The general geometric formulas are

```
chi_dot = 2M/(3a^2)-2*a1/a-4*b1/(3a),
gtilde_rad_dot = 8M/(3a^2)-4*b1/(3a),
gtilde_tan_dot = -gtilde_rad_dot/2,
P_dot = 3/a^2*(a1+M/a), Theta_dot=0 for these vacuum initial jets.
```

For xi>=0, cancelling only the shift boundary rate and dt N_raw with the initial quadratic null condition would require

```
eta*a^2/S = 2*(1+xi*a)/(3+xi*a).
```

The dimensionless right side lies in [2/3,2), so eta10 is impossible for any nonnegative xi at S1,a.5, where its left side is 2.5. Eta5 would require xi=14/3 and jets -.6,+1.6; that still does not satisfy the lapse boundary-value gate.

On the smooth finite-trace, Theta=0 vacuum manifold, the pinned shift implies `P=-3*beta_ref*Omega'/alpha`. Substitution into the physical-P lapse pole gives exactly

```
S_alpha=-(alpha-alpha_ref)*[(3*alpha+alpha_ref)/a
                          +xi*(alpha+alpha_ref)].
```

With positive lapse and xi>=0, finite lapse RHS therefore also pins `alpha_scri=alpha_ref` and requires alpha_dot_scri=0. The leading numerator derivative is independently checked as

```
dt S_alpha = -alpha0^2*P_dot-alpha0*(2*xi+1/a)*alpha_dot
             +alpha0/a*beta_dot.
```

Requiring these gauge boundary values, dt N_raw=0, and the original initial quadratic null falloff yields the original geometric first jets and the unique rate pair

```
a1=-M/a, b1=2M/a, xi=1/a, eta=S/a^2.
```

At S1,a.5 this is xi2/eta4. Chi/gtilde/P leading rates vanish as a consequence of this solution for the fixed initial data; their vanishing is not imposed on arbitrary live fields, nor are A/Lambda pinned by hand. The trace/Q/Theta assumptions are stated for this constraint-satisfying BH initial case, not as an arbitrary live falloff.

Even xi2/eta4 with the original complete geometric jets gives `dt N_raw=-36*Omega+...`. Exact rational second-jet algebra gives `dt N_raw=[-36+48*delta_b2]*Omega+O(Omega^2)`, independent of delta_a2 at this order. Adding `tau*(3/4)*Omega^2` to the initial radial shift cancels this next coefficient while retaining alpha_dot=beta_dot=chi_dot=gtilde_dot=P_dot=Theta_dot=0 at scri. The full kernel records this zero-jet pair and second-jet example. They pass only these instantaneous leading/next conditions; no preserved dynamic manifold or BH evolution is established.

## Reproducible kernel and limiting checks

`run_audit.py` runs seven checks: the actual conformal tensor/gauge kernel in Release and Debug with AddressSanitizer/UndefinedBehaviorSanitizer, independent general first-jet algebra, exact rational second-jet algebra, the leading finite-pole obstruction, the balanced zero-jet pair second gate, and a 100-digit factored outer oracle. All pass; test time is 3.4493 s and total including builds 5.4896 s. The source parent is `b0313c43c8f1af6fc9e3315a75a072694d14ec67`. All critical production and scratch sources remained unchanged during this audit.

Each kernel executable records 104 rows: four source choices, eight radii including exact scri and machine-scale Omega, and three initial gauge choices, plus eight balanced xi2/eta4 controls. Its raw full kernel and factored derivative agree to 1.61e-10 for Omega>=1e-4. Fixed ADM H/M remain <=1.95e-14/8.88e-16 under the gauge-only changes. Modified lapse/shift Cartesian first/second derivatives are checked at six oblique collar radii with finite-difference scaled discrepancies 3.59e-8/1.94e-4. A 101-point positive-lapse scan through the collar has minimum alpha=1.84470; the earlier positive core/throat is unaffected.

The factored exact-scri results agree with the symbolic limits. Raw floating assembly is not trustworthy at machine-scale Omega: for example at Omega~2e-15 its source-off, preferred and eta10 null rates are approximately 35.26, 13.04 and -22.52, versus the factored 24, 0 and -56. This roundoff discrepancy is preserved in the logs; the small-Omega conclusions use explicit factoring checked against the high-precision oracle, not direct denominator substitution. Production `AssembleGaugeInterior` only assembles its supported alpha pole; the audit explicitly includes every added shift pole, matching the private candidate assembly.

`receipt.json` contains all commands, durations and hashes; Release/Debug JSONL logs record both raw and factored rates. The main audited kernel SHA256 is `a9b1c430d05e138154b86b2aec8a4cfcb116c4757b43f1819592fcfa406ca497`; the first-/second-jet proofs are `ab37c59d6aa4b8d0b7f946c053b8ffb7a190f2486fd66297355611dc5eef1020` and `2bd2b81584f11027144ff62ac49d6fd92161c3e842445fac307aeeeb50381035`. All work remains ignored scratch, with no runtime integration or BH evolution.
