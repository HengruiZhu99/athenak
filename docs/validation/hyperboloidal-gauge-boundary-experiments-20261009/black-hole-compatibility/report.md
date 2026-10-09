# Detached wormhole height: reviewable scratch implementation

This construction admits M=.5 while preserving the actual Minkowski reference S=1, a=.5, compactification/height layer (.05,.95). It chooses a separate BH height cutoff (.30,.95) and reconstructs the BH metric and connection from that height. The initial vacuum constraints, consumed analytic jets and exterior stationary geometric kernel pass; there is no BH evolution acceptance or integration. The full finite-pulse Minkowski stability gate remains outstanding.

## Geometry and throat condition

Keep the production monotone `Omega(r)` and `L=Omega-r*Omega'>0`. Let `R=r/Omega` be the physical isotropic radius, `m=M/(2R)=M*Omega/(2r)`, `psi=1+m`, `N=(1-m)/(1+m)`. Select an independent smooth height weight `W_B`, exactly zero through compact radius `rB0` and exactly one after `rB1<S`, with

```
R(rB0) > M/2,
b_B=r*W_B/a, A_B=sqrt(Omega^2+b_B^2), v_B=b_B/A_B.
```

On the exterior where the boost is active, the Schwarzschild static height is `dh/dR=psi^2*v_B/N`. Its pulled-back conformal spatial metric is

```
bar_gamma_rad = psi^4*L^2/A_B^2,
bar_gamma_tan = psi^4,
chi_B=(A_B/L)^(2/3), chi=chi_B*psi^-4,
gtilde_rad=chi_B*L^2/A_B^2, gtilde_tan=chi_B,
Lambda=Gamma(gtilde).
```

Here “rad/tan” are Cartesian radial/tangential eigenvalues; the angular spherical component includes r^2. These gtilde/Lambda fields belong to the BH height and generally differ from the actual Minkowski reference. On the exact height core, b_B=0, beta=0 and K_ij=0. The metric there is precisely the time-symmetric Schwarzschild wormhole Cauchy slice pulled back through the monotone R(r). Omega need not equal one in that larger BH core. At the throat R=M/2 the physical three-metric, curvature and all consumed jets are smooth, and the lapse below is positive. The static four-dimensional chart has a degenerate lapse at the bifurcation sphere; it is used only to derive exterior slice geometry. The evolved initial lapse is separately prescribed.

For the audited choice, the compact throat is `r=.24940753251830589`; the BH height core ends at physical isotropic `R(.30)=.30267991845762615`, safely outside M/2=.25. A possible mass-dependent default is `rB0=max(rOmega0,.6*M)` and `rB1=rOmega1`, provided `rB0<rB1` and the exact throat condition is validated. The production admissibility Omega<=1 then ensures R(rB0)>=rB0>M/2. This rule gives .30 for M=.5 and recovers the actual reference height as M tends to zero. Its dependence on M has a harmless parameter kink, while each spatial profile remains smooth. A fixed .30 cutoff instead tends to a different Minkowski height as M tends to zero; it does not tend to the original reference height.

## Curvature and initialized gauge

With the existing extrinsic-curvature sign convention, the physical radial/tangential eigenvalues and stored physical trace are

```
k_r=psi^-2*[(-Omega*b_B'+b_B*Omega')/L
             -2*(b_B/r)*m/(1-m^2)],
k_t=-psi^-2*(b_B/r)*N,
P=k_r+2*k_t.
```

Evaluate the exact b_B=0 core branch before any factor 1/(1-m^2). The factored physical shear divided by Omega is

```
D_B=(-b_B'+b_B/r)/L,
S_B=(k_r-k_t)/Omega
   =psi^-2*[D_B-b_B*M*(2-m)/(r^2*(1-m^2))],
A_rad=2*gtilde_rad*S_B/3,
A_tan=-gtilde_tan*S_B/3.
```

This form avoids forming a small physical shear difference and dividing it by Omega. The scratch implementation computes P/A and their first spatial derivatives; their second derivatives are not consumed by the current kernels and are not advertised as analytic jets. It returns second derivatives of alpha, beta, chi and gtilde, and first derivatives of Lambda, P and A in the native `Z4cJet<T>` format.

The initialized shift is the stationary exterior shift, extended by zero through the exact core. A positive conformal lapse blends a usual pre-collapse physical lapse in the core to the geometric lapse in the outer collar:

```
beta^r=-N*psi^-2*b_B*A_B/L,
alpha_init=A_B*[(1-W_B)*psi^-2+W_B*N].
```

N can be negative on the inner asymptotic end, but W_B is exactly zero there. Where W_B>0, the enforced throat condition gives N>0, so alpha_init>0. At the audited throat alpha_init=.24940753251830591. As r tends to zero, alpha~4*r^2/M^2 and chi~16*r^4/M^4; the origin branch avoids every r division and returns their finite limits (including lapse Hessian 8/M^2). Native moving-puncture grids must still exclude the exact origin with even cell counts. This positive lapse is deliberately dynamical in the core/height transition; its initial geometric RHS is about .95523, not zero. The Minkowski gauge/reference source is separately dynamical for BH data. Neither is subtracted.

## Outer compatibility and vacuum identities

Once both height and compactification have reached their outer branches, the height is mass corrected:

```
dh/dR=1+2*M/R+O(R^-2),
h(R)=R+2*M*log(R)+O(R^-1),
bar_g^{rr}=Omega^2*psi^-4/L^2,
bar_g^{ab} grad_a(Omega) grad_b(Omega)
    =Omega^2*psi^-4*(Omega')^2/L^2.
```

Thus the surface Omega=0 is null; directly applying the Minkowski height to Schwarzschild would miss the O(1/R) mass term and produce the known spatial-metric pole. Exact-scri initial values are alpha=S/a, beta^r=-S/a, chi=1, gtilde=I, Lambda=0, P=-3/a, A_rad=-4*M/(3*a*S), A_tan=2*M/(3*a*S). The finite A limit encodes the mass-dependent physical shear falloff and must not be set to the Minkowski A=0.

`independent_geometry.py` derives the spatial metric from the height-transformed Schwarzschild four-metric and independently checks the curvature from the stationary ADM definition. For physical `gamma_rr=E`, areal radius rho=psi^2*R and the eigenvalues above it proves symbolically, for arbitrary Omega and b_B,

```
H=R3+4*k_r*k_t+2*k_t^2=0,
M_r=-2*k_t'+2*(rho'/rho)*(k_r-k_t)=0,
rho/2*(1-rho'^2/E+rho^2*k_t^2)=M.
```

The proof also evaluates the exact time-symmetric core scalar curvature directly, without dividing by rho' or N at the throat. At that throat rho=2M, rho'=0, K=0, the null expansions vanish and the area is 16*pi*M^2. For M=.5, rho=1 and area=4*pi. The Schwarzschild Kretschmann value there is 48*M^2/rho^6=12; this value follows from the identified Schwarzschild geometry, rather than a separate numerical four-dimensional Riemann extraction.

With exterior geometric conformal lapse `alpha_static=N*A_B` (positive only on the tested exterior), the unsubtracted actual `ConformalRHS` vanishes to floating accuracy. This is a check of the metric/curvature/lapse/shift jets and sign conventions, not a fixed point of the actual Minkowski reference gauge. No BH RHS subtraction is introduced anywhere.

## Audits and limits

`run_audit.py` saved commands, durations, binary/source SHA256 and logs in `receipt.json`. Nine checks passed: four Release executables, the same four in Debug with AddressSanitizer and UndefinedBehaviorSanitizer, plus the independent symbolic proof. Tests took 5.0515 s; builds plus tests took 12.8013 s. macOS LeakSanitizer is unsupported and disabled; address/undefined checks remained enabled. All critical production and scratch source hashes were unchanged during the run. The recorded parent HEAD is `2564570bf43aef7e8bcb2216bf0a2a9ba4ca2717`.

| Audit | Result |
|---|---|
| M=.5 initial constraints, 542 axis/oblique points | max H 2.04e-14; max M 3.50e-15 |
| Independent geometric Misner–Sharp mass from initial fields | max error 4.74e-10 |
| Determinant / tracefree residuals | 8.88e-16 / 8.02e-16 |
| Complete consumed 21-field Cartesian derivatives at 9 oblique radii | first scaled error 9.55e-11; second 4.49e-6 |
| M=1e-9 with matching height, all 21 values vs original reference | max difference 2.00e-7 |
| Exterior static geometric RHS, 284 points | max 4.74e-11, Omega>=approximately 2e-4 |
| Counterfactual detached-height Minkowski geometry / physical-trace gauge fixed point | 6.03e-12 / 4.44e-16 |
| Generic endpoints, small Omega and cutoff-underflow tails: 792 points, 4 configurations | H 6.44e-14; M 4.13e-15; finite consumed jets |
| Exact-scri field limits | residual 2.22e-16 |
| Invalid mass, disabled height and unsafe throat placement | all 6 rejected |

The generic audit changes S, a, M and the independent height endpoints, tests both compactification/height endpoints and representable inner derivative tails through exponential underflow, and checks the exact origin branch. The actual M=.5 Lambda differs from the actual Minkowski Lambda by as much as 1.75; this guards against accidentally reusing the reference connection.

At Omega~2e-15, floating cancellation in the existing `regular+pole/Omega` assembly produces an exterior stationary RHS as large as 9.78 even though all analytic jets and regular/pole blocks are finite and the exact-scri pole numerator vanishes. A finite numerator alone is not an accurate l'Hopital closure. This scratch data construction does not solve the existing very-small-Omega assembly problem or justify using an unsubtracted BH background on ghost nodes without a consistent limit treatment.

## API and implementation boundary

```
LayerReference<T> actual_reference(S,a,{true,rOmega0,rOmega1});
DetachedWormhole<T> data(actual_reference,M,{true,rB0,rB1});
Z4cJet<T> u=data.At(x,y,z);
```

`DetachedHeightGeometry<T>` provides the counterfactual mass-zero spatial geometry used by `DetachedWormhole<T>`; it must never replace `CartesianConformalPatch.reference`. Runtime integration would expose independent BH height endpoints, validate the exact throat condition and exclude the origin, then use these full jets only for analytic initial-data reconstruction/deviation differentiation. Every gauge/reference source and any existing roundoff subtraction must keep the actual Minkowski reference. Integration and BH evolution are deferred until finite-pulse Minkowski acceptance.

Source SHA256:

* `detached_height.hpp`: `31a18e8bbcd5aa82f3359ffb0c7e319418a4f8619d83269af5fe5a87a45c3dfd`
* `detached_wormhole.hpp`: `d4123cbebf780fa9fbf0b2b3f6b37f2dfa701c73454c5cb14326600ce332e5b3`
* independent four-metric/ADM proof: `b869f9cf845483ed6410494c7c31e86fd41b6cbc56023335ef49c19dd4b04004`

Binary and remaining audit hashes are in `receipt.json`; all files in this directory are ignored scratch artifacts, with no production edits.
