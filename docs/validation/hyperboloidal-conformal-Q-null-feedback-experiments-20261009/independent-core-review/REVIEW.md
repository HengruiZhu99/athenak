# Independent conformal-Q/null-feedback review

This is a read-only mathematical and source review of the separate scratch
candidate. It does not rerun the actual tensor gate or launch any propagation,
native evolution, or black-hole test. The final receipt pins the reviewed
immutable local index and its helper bytes. Production equations remain those
of implementation `27c19d20696ea6dd4704032c51dfd026218f64f2`.

Reviewed core index: `dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96`.
The frozen source/math has no correction required within its stated scope.
One prose typo is clarified externally: DERIVATION's weighted gradient formula
must read `alpha^2 dlog(alpha/h)=alpha dalpha-alpha^2 dlog(h)`; its actual
`alpha2du` implementation already evaluates this correct relative gradient.
The frozen original is preserved.

The source signs and factorizations are consistent with the future-normal
convention, stored `P=K_phys-2 Theta_phys`, and
`Q=(P-3 omega_n)/Omega`. The null scalar is separately named
`Nraw=chi*gtilde_inverse^{ij} Omega_i Omega_j-omega_n^2`.
Write `h=alpha_ref`, `B=beta.dOmega`, `Bh=beta_ref.dOmega`, and
`D=alpha*Bh/h-B=alpha*(omega_n-omega_n_ref)`. The Q lapse numerator is

```
-alpha*(alpha+2*(1-W))*(P-P_ref)
  +3*(alpha+2*(1-W))*D.
```

It is algebraically the original Q prescription without forming a Q quotient.
The near-reference difference for D and the direct collapsed-lapse form are
equivalent. The direct branch retains a representable small alpha when
`alpha-h` rounds to `-h`. Likewise

```
alpha^2*deltaNraw = alpha^2*deltaG + D*(B+alpha*Bh/h)
```

removes avoidable live-lapse divisions from the actual feedback. The preferred
projection evaluates `alpha^2*Delta` directly, and the spatial logarithmic
gradients are evaluated as weighted products. These are finite-lapse numerical
repairs. They do not establish uniform hyperbolicity, a bounded generalized
harmonic source, lapse positivity, or a limit through alpha=0. The unweighted
`NullDifference` diagnostic still divides by the live lapse; the gauge helper
does not use it to evaluate the feedback.

An independent four-metric Christoffel contraction gives, for gauge-rate
changes at fixed spatial geometry,

```
delta Gamma4^0 = -delta alpha_dot/alpha^3
delta Gamma4^i = beta^i*delta alpha_dot/alpha^3-delta beta_dot^i/alpha^2.
```

Thus the added beta pole has the advertised sign:
`delta F^i=-V*sigma*Omega_i*deltaNraw/(|dOmega|^2*Omega)` and
`delta BoxOmega=+V*sigma*deltaNraw/Omega`. At W=1,
`Gamma4^a+2Z4^a=F^a` and

```
F0=(beta.grad(log(alpha_ref))+nu*log(alpha/h))/alpha^2-Kbar_ref/alpha
BoxOmega=Omega*What_ref+2Z4^i*Omega_i+V*sigma*deltaNraw/Omega.
```

Here `Z4^0=Theta_phys/(alpha*Omega)` and
`Z4^i=chi*gtilde_inverse^{ij} Z_j-beta^i*Theta_phys/(alpha*Omega)`.
The contracted temporal contribution is retained. A finite F0 at positive
lapse does not imply bounded F^i for arbitrary off-null states: the feedback
source still contains `deltaNraw/Omega`.

The optional alpha-only blend uses `(1-W)*physicalP_alpha+W*Q_alpha` for
both regular and pole parts, while retaining the Q/preferred spatial extension.
It inherits `scri_lapse_damping`; the intended native setting is xi=1/a, with
xi=1.5 retained as a control. Exact W=0 short-circuits the unused Q branch.
In a transition region, a changed lapse rate A changes the effective source by
`delta F0=-A/alpha^3`, `delta Fi=beta^i*A/alpha^3`, and
`delta BoxOmega=-(beta.dOmega)*A/alpha^3`. Consequently the displayed preferred
Box identity applies at W=1; the transition keeps an algebraic extension and
does not silently inherit that identity. All changes are algebraic in live
values, fixed reference jets, and fixed weights, so the complete principal
symbol is unchanged. This does not bound lower-order finite-radius behavior.

The geometric Cauchy core r<=.05, gauge W=0 region r<=.45, gauge transition
.45<r<.85, and feedback transition .85<r<.95 are distinct. In the exact
Omega=1 core, dOmega=0 and constant alpha_ref make the Q and physical-P lapse
sources identical. Future puncture or wormhole-to-trumpet matching is not
admitted by these local Minkowski tests.

The independent read-only check reconstructs all eight sigma=5 reference outer
pole matrices rationally. Its maximum reconstruction error is
3.552713678800501e-15. For each, rank(M)=rank(M^2)=11, so the nine zero
eigenvalues are semisimple. With K=kappa_input*a^2 and z=a^2*lambda, the last
scalar cubic is

```
z^3+(2K+10)z^2+(20K+11)z+28K-30.
```

Its Routh determinant is `40K^2+194K+140`; together with the other linear and
quadratic factors this gives 11 strictly negative-real-part roots precisely
when K>15/14 for sigma=5. All sampled a=.5,.75,1,2 and kappa_input=5,10 satisfy
that condition. Exact algebra applies to the reconstructed analytic matrices,
with the discrepancy from the raw floating matrices explicitly retained.

The Einstein gauge witness has `delta alpha=Omega`,
`delta beta=-Omega*n`, and all geometric/P/Theta deviations zero. Using the
full radial alpha_ref, its initial perturbations are
`deltaNraw=-a*Omega^3+O(Omega^4)` and
`deltaQ=(3a^2/2)*Omega^2+O(Omega^3)`. Independent radial equations give

```
(alpha_dot0,beta_dot_n0,P_dot0,chi_dot0,gtilde_nn_dot0)
  =(1/a^2,0,3/a^2,-2/(3a),4/(3a)).
```

Both `Nraw_dot0` and `(P-3omega_n)_dot0` vanish. This is a condition at the
initial corner. It is not a preserved first/second-jet manifold, PDE closure,
or a demonstration of amplitude blowup in the alternative equation. The
prior physical-P candidate with its fully rederived preferred projection and
sigma feedback already has the same two cancellations: writing d=1+2xi*a,
its lapse/shift rates are `(-d/a^2,(d+1)/a^2)`, with the same sum. The new Q
candidate changes the distribution of those rates; this witness does not
establish a unique repair or comparative improvement. The retained finite-Q
counterexample further prevents a claim that arbitrary finite Q alone closes
the smooth corner hierarchy: for `deltaP=q*Omega` with every other field equal
reference, the independent radial equations give
`alpha_dot0=P_dot0=-q/a^2`, `Theta_dot0=-2q/a^2`, and
`(P-3omega_n)_dot0=2q/a^2`.

The local admission covers the reference outer value pole, complete principal
symbol, nonlinear finite-Omega source/factoring/reference identities, both xi
choices in the alpha blend, and the stated initial scalar corner. The receipt
does not contain a finite-Fourier transition screen. No global, native,
black-hole, energy-transfer, uniform puncture, or evolution-preserved scri
regularity conclusion follows from this review.
