# Next boundary candidate: physical-constraint radiation, then a full-system lift

The next justified candidate is a **coupled radiative boundary operator on the physical constraints**, first derived and tested in a linear spherical full-state model. The annulus result does not justify adding `−u/r` to the existing incoming full-state characteristic amplitudes. A direct replacement of Theta and Gamma RHS values also conflicts, in general, with retaining both existing incoming gauge conditions. There is a concrete differential boundary operator below, but a compatible discrete full-system implementation remains to be derived. No AthenaK source or job was changed for this audit.

## Primary-source context

Ruiz, Hilditch and Bernuzzi formulate boundary conditions on Theta and Z, then derive compatible metric/gauge boundary evolution. Their Appendix C contains a coupled spherical implementation, including the flat-background `Theta_t=−Theta_r−Theta/r` condition. Its companion Gamma, Khat and A equations contain additional radial terms; it is not a per-variable Sommerfeld prescription. Their well-posedness analysis uses frozen coefficients, while the full-system analysis has a restricted spherical/gauge scope. Constraint damping is omitted from that paper's principal analysis. These distinctions matter for the present nonconstant trumpet, damping, Cartesian variables and adapted gauge. [Primary paper, arXiv:1010.0523](https://arxiv.org/pdf/1010.0523), Secs. II F, IV and Appendix C.

The paper's spherical Gamma variable cannot be copied into a Cartesian code as an ordinary vector transformation: conformal connection functions contain a metric-defined connection term. The difference `Q=Gamma_evolved−Gamma_metric=2gtilde^{-1}Z` does transform as a vector and supplies a safer derivation route.

## What the existing boundary actually enforces

In `src/z4c/z4c_Sbc.cpp`, `zero_rate` takes the current full-state conformal frame and gauge coefficients, forms incoming characteristic combinations of residual momenta and residual metric/gauge derivatives, and sets their coefficient-frozen RHS combinations to zero. The four scalar rows are two gauge rows followed by two light-speed rows labeled constraints. In a linear spherical perturbation of the stationary `R0=M=1` background, the latter are

```
C1 = alpha*Theta + chi*Gamma/2 + u'
C2 = 4k/(3alpha) + 2Theta/(3alpha) − 2A/alpha − Gamma + h'
```

Here `u=delta chi`, `h=delta gtilde_nn` with tangential metric perturbation `−h/2`, `k=delta Khat`, `Gamma=n_i delta Gamma_evolved^i`, and `A` is the trace-free radial projection of `delta Atilde_ij` used by the code. Primes are Cartesian radial derivatives along a ray. All coefficients in this linear map are background values: `alpha=r/(r+1)`, `chi=alpha²`.

These C1/C2 amplitudes are **not** Theta or Q, and they need not vanish for a constraint-satisfying metric/gauge perturbation. The physical incoming constraint waves involve derivatives of C1/C2 plus background and spherical lower-order terms. Therefore `C1_t=C2_t=0`, `C1_t=−C1/r`, and `Theta_t+(c−b)Theta_r=0` are three different boundary conditions.

The current implementation modifies the four scalar momentum RHS entries `(k_t,Theta_t,A_t,Gamma_t)` through a local4×4 solve; it retains the already computed metric and gauge RHS derivatives. The two vector sectors and two tensor radiation rows have their own maps. A future constraint change must retain the prescribed incoming gauge and radiation data as separate data, not absorb them into a residual damping rule.

## Physically weighted spherical radiation candidate

Let

```
R = r+1                 (background areal radius)
b = r/R²               (radial shift)
c = alpha*sqrt(chi) = alpha²
v = c−b                (outgoing coordinate light speed)
k_out = partial_t + v partial_r
```

One leading outgoing-amplitude prescription is

```
k_out(R*Theta)=0,
k_out(R*Z_hat)=0,
Z_hat = sqrt(chi)*q/2,   Q^i=q n^i.
```

The second expression uses the physical orthonormal radial projection of Z. Since `R*sqrt(chi)=r` on this trumpet, the conditions become

```
F_Theta = Theta_t + v*(Theta' + omega_Theta*Theta)=0,
F_q     = q_t     + v*(q'     + omega_q*q)=0,
omega_Theta = 1/(r+1),   omega_q = 1/r.
```

This is a geometrically motivated leading radiation approximation, not an exact absorbing condition for the coupled curved-background constraint waves. It includes basis/area factors and is homogeneous on the physical constraint surface. The earlier annulus `p=1` experiment instead used `omega_Theta=omega_q=1/r`; the two prescriptions agree only asymptotically. Neither prescription establishes a 3D Cartesian-face or corner operator by itself.

A separate, cheap spectral check confirms that the benefit survives this weight change in the same continuum annulus `0.2M<r<4M`:

| Damping | Largest real eigenvalue, degree64 | Degree128 | Positive finite eigenvalues at128 |
|---|---:|---:|---:|
| `sigma=0.1alpha` | −0.0727906934463/M | −0.0727906934460/M | 0 of514 |
| `sigma=0.3` | −0.0045706882747/M | −0.0045706882726/M | 0 of514 |

All finite generalized eigenvalues were inspected. The two infinite values represent algebraic boundary rows. This establishes convergence of these finite spectra, not a continuum energy estimate, absence of transient growth, or full geometric/gauge stability. These are separately labeled in `radial-areal-weight-comparison.json` and `annulus-areal-*.json`; no AthenaK evolution is represented by those numbers.

## Exact linear spherical full-state lift

The following relations apply to the **current Z4c geometric equations**, `kappa2=0`, and either `sigma=alpha*kappa1` or a chosen bounded coordinate product. A coupled covariant-source formulation would require rederiving the constraint equations and this lift; it is not covered automatically.

Algebraic constraints imply

```
Gamma_metric^r = h' + 3h/r,
q = Gamma − h' − 3h/r,
delta Atilde_rr = A − 2K*h/3,
j = delta K^r_r = A + 2K*h/3 + (k+2Theta)/3,
K = 1/R².
```

The physical linear momentum constraint is

```
m = A' − (2/3)k' − (4/3)Theta' + 3A/R
    + 2K*h/(3R) + (5/3)K*h' + 2K*(u/chi)'.
```

The Hamiltonian is `H=delta R_phys+4K*j`, with

```
delta R_phys = chi*(h''+5h'/r+3h/r²) + 2u''+4u'/r
               −2h*chi''−2chi'*h'−4h*chi'/r
               −5chi'*u'/chi
               +(5/2)h*(chi')²/chi +(5/2)u*(chi')²/chi².
```

These maps have been independently checked symbolically against the scalar curvature of the physical warped spherical metric and the covariant momentum divergence; they are not guessed principal-only formulas.

Using the audited constraint evolution to eliminate time derivatives gives the two boundary residuals in physical constraints:

```
F_Theta = alpha*H/2 + c*Theta'
          + alpha*chi*(q'+2q/r)/2
          + (v*omega_Theta−2sigma)*Theta,
F_q = 2alpha*m + 2alpha*Theta' + c*q'
      + (v*omega_q−2sigma)*q.
```

They vanish identically for `H=m=Theta=q=0`, independent of any nonzero physical metric/gauge perturbation. This is the preservation property that an arbitrary `−C/r` source lacks.

In terms of the code's incoming light-speed amplitudes, there are exact identities

```
F_Theta = alpha*C1' + L_Theta,
F_q     = −chi*C2' + L_q.
```

Every coefficient in the lower-order terms is explicit. With `R=r+1`,

```
L_Theta = 5u/(rR³)
          + r*(9r²+9r+1)*h/(3R⁵)
          + 2r*k/(3R³)
          + [v*omega_Theta−2sigma+r/(3R³)]*Theta
          + 2r*A/R³ + r³*q/R⁴
          + (2r−3)*u'/R² + r²*(r−2)*h'/R⁴,

L_q = −8u/(r²R²)
      + (9r²+22r+9)*h/(3R⁴)
      −4k/(3R²) −2Theta/(3R²)
      +2*(3r+1)*A/R² + (v*omega_q−2sigma)*q
      +4u'/(rR) −r*(9r−1)*h'/(3R³).
```

Thus the physical radiation conditions supply **normal-derivative boundary data**

```
C1' = −L_Theta/alpha,
C2' = L_q/chi.
```

They are a concrete linear boundary operator, including damping and all trumpet/spherical coefficients. The source identities and their checks are saved in `derive_boundary_lift.py` and `boundary-lift-identities.json`.

## Why this is not a drop-in replacement for zero_rate

Directly enforcing the two primitive radiation equations would prescribe

```
Theta_t = −v*(Theta'+omega_Theta*Theta),
Gamma_t = (h_t)' + 3h_t/r − v*(q'+omega_q*q).
```

The metric RHS `h_t` is unchanged by the current local momentum correction. Retaining both existing incoming gauge rows while imposing these two equations produces four conditions whose coefficient matrix in `(k_t,Theta_t,A_t,Gamma_t)` has an identically zero A column and rank at most3:

```
[ gauge_k0       0       0       0       ]
[ gauge_k1   gauge_Theta 0    gauge_Gamma ]
[     0          1       0       0       ]
[     0          0       0       1       ]
```

It is generically inconsistent. This is an obstruction to that particular **local RHS overwrite**, not proof that the full differential IBVP is overdetermined. The missing condition acts through normal derivatives of A and of metric data. The normal-derivative identities above provide the proper route to a higher-order coupled boundary system.

A consistent implementation must use the full characteristic evolution, including background-coefficient derivatives and lower-order sources, to turn those derivative data into compatible evolution/SAT/Bjørhus boundary operators. One cannot simply replace `target_rate[2/3]` by `−C1/2/r`: that neither imposes the derived F conditions nor preserves arbitrary physical constraint-satisfying perturbations. A tentative projected correction must be tested by substituting the **corrected** full RHS into Theta_t and Q_t, including time derivatives of metric-defined Gamma, rather than only checking its own local enforcement diagnostic.

At nonlinear order the conformal frame and map also evolve. Their time derivatives, algebraic projection, gauge dependence, and full-minus-background treatment must be included. On Cartesian faces, angular/tangential derivatives and face normal versus radial directions are additional required terms. None is represented by a universal facewise `1/r` multiplier.

## Concrete next validation sequence

1. Build a separate linear spherical **full-state** operator using the actual adapted lapse/shift gauge and current geometric equations. Impose the existing incoming gauge data plus the two derived normal-derivative constraint boundary relations. Keep physical/outgoing data distinct. In spherical symmetry there is no TT radiation sector to validate yet.
2. Derive the discrete boundary closure with compatible derivative operators. Verify the induced constraint evolution equals the intended annulus boundary equations; check the full constraint map, including metric-defined Gamma, its time derivative and all lower-order terms. A constraint-only spectrum is insufficient for this gate.
3. Check background-zero cancellation, constraint-satisfying mass/gauge perturbations, controlled outgoing constraint pulses, and boundary-reflection coefficients against a larger-domain reference. Establish resolution convergence and repeat at multiple boundary radii, damping products and inner cutoffs. Inspect nonnormal transient growth as well as eigenvalues.
4. Only then extend to full 3D normal/tangential geometry, preserve incoming gauge and two radiation polarizations, and prove/check edge and corner compatibility, immutable stencil inputs, MPI exchange, and GPU execution. Retain the selected ghost-fill closure as a separate numerical choice until jointly validated.
5. Before matter evolution, test the actual matter constraint sources and boundary assumptions. The present derivation is vacuum; it does not waive the existing CPBC matter-energy ceiling or establish valid boundaries for stellar debris crossing the outer box.

This provides a focused boundary hypothesis and a checked linear lift, with an explicit implementation obstruction. It is not yet an evolution-ready 3D boundary patch, and no production setting should be changed on its basis.
