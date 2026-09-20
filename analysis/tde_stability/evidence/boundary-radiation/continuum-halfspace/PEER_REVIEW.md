# Independent review of the continuum half-space test

Read-only review of `determinant.py` and the saved initial real-axis scan. No source equations or evolution jobs were changed.

## Boundary equations and normal-mode convention

The ansatz is `exp(lambda*t+s*x+i*ky*y)` on the interior half-space `x<0`, with outward normal `+x`. Therefore decaying normal modes have `Re(s)>0`. The script uses that sign, and it requires ten such roots rather than accepting a partial basis. The q/p block elimination, quadratic companion signs, and reconstruction `p=C^{-1}(Q0+s*Q1)q` agree with the full twenty-field volume equation. Full-state bulk eigen-residuals are recorded.

For the current nonsingular p-only boundary map, smooth consistency means its momentum-RHS correction vanishes. In the scalar sector this is equivalent to vanishing incoming lapse/shift rates and `FTheta=FQ_n=0`. In each transverse sector the vanishing Gamma correction gives the gauge rate condition; the remaining A correction then gives `FQ_A=0`. The TT correction similarly gives its incoming rate condition. Thus the script tests the correct continuum boundary conditions associated with the v2 modified-RHS construction.

For a volume mode and constant background, the discrete-helper distinction disappears: the exact metric-defined Gamma time derivative equals lambda times the metric-defined Gamma. The physical constraint radiation factor is consequently

```
lambda + (c-beta_n)*s - i*beta_A*k_A,
c=alpha*sqrt(chi).
```

The script uses beta_A=0, hence the corresponding simpler factor. A nonzero constant tangential shift translates the imaginary temporal frequency; it does not change the real growth rate. This is an explicitly normal-shift-only comparison, not a curved-background model.

The gauge and TT rate equations can be divided by lambda when `Re(lambda)>0` to test roots. This division must not be used to claim a uniform estimate as lambda approaches zero, since the rate-to-data normalization then changes.

## Initial scan and conditioning caveat

The initial real-axis scan has 27 samples per boundary/damping combination. Its smallest row-scaled singular values are 3.8119e-4 for undamped radiation and 4.8526e-3 with the selected damping, both at lambda=1.2 and ky=2*pi. No sampled value is an exact zero. This is a bounded rank scan, not a proof of boundary stability and not a search of the complete complex half-plane.

Raw decaying-mode eigenvectors become badly conditioned: the largest recorded condition number is about1.77e10. Orthogonalizing their traces afterward cannot recover generalized modes or precision already lost by diagonalizing a nearly defective normal symbol. Near `lambda=beta_n*abs(ky)`, the normal roots coalesce at `s=abs(ky)` in the principal system. That is precisely where polynomial-times-exponential generalized solutions can matter. Consequently neither an apparent root nor a small positive singular value should be interpreted there without a stable invariant-subspace check.

The suggested bounded verification uses an ordered complex Schur/QZ stable subspace of the first-order companion, then evaluates the boundary functional on its full normal jets. If `T` is the companion and `Z` its stable Schur columns, reconstruct `u=J Z`, `u_x=J T Z`, and `u_xx=J T^2 Z` using the same q/p elimination map J. This retains generalized decaying solutions without dividing by eigenvector differences. Values of lambda=1.2 and lambda=beta_n*ky are sufficient first checks; a broad scan is not needed for this review.

## Ordered-Schur verification and positive root

The follow-up `schur_check.py` implements that stable-subspace construction correctly. I reviewed the companion block signs, Schur sorting, reconstruction of p and its derivative, and `dQ=Q*T`; these retain generalized normal modes. Its stable trace basis is well conditioned at the candidate (condition number about41.6), and the invariant-subspace defect is about7.8e-15.

I also reviewed `validate_root.py`. Its coordinate chart removes arbitrary Schur phases, brackets a real determinant zero, constructs the boundary nullvector, and evaluates the original twenty-field differential equations along a smooth decaying profile. At ky=2*pi with the selected damping it finds

```
lambda = 1.2307116366820878 / M
maximum relative bulk-equation defect = 7.84e-14
relative unscaled boundary defect = 1.66e-16
Theta peak = 8.01e-15
Q peak = 2.55e-14
```

The state is normalized to unit maximum amplitude at the boundary. Its norm at x=-2 is about1.53e-5 of its boundary norm. The old zero-rate boundary matrix has a smallest normalized singular value about0.140 at this same frequency, so it does not admit this particular mode.

Without lower-order damping, roots at ky=pi, 2*pi and4*pi are respectively0.5987161270, 1.1974322540 and2.3948645081. Their linear scaling with tangential frequency is the expected dangerous principal-boundary behavior. These are validated positive roots, not just small singular values from the original ill-conditioned eigenvector scan. A nonzero constant tangential shift translates their imaginary frequency without changing their positive growth rate.

The supported conclusion is therefore stronger than the initial scan allowed: the v2 combination of physical constraint radiation with the existing zero-rate gauge/TT conditions admits an unstable **frozen continuum half-space mode**. Improving ghost polynomial order or the inner derivative accuracy cannot by itself remove this continuum incompatibility. Since Theta and Q are essentially zero in the mode, its interpretation is not simply an injected incoming constraint wave.

The additional `physical_and_gauge.py` check gives H and M maxima about3.43e-14 and6.95e-14, but electric/magnetic linear Weyl norms about0.145 and0.0103. I checked these formulas independently: E is chi times the linear physical Ricci in the orthonormal background frame; the trace is the Hamiltonian constraint. B is the physical derivative curl of `A_ij+delta_ij*(Khat+2*Theta)/3`, with the correct sqrt(chi) derivative factor. Quadratic background-curvature terms vanish in this frozen flat model. Nonzero E/B exclude a pure coordinate disturbance. The supported description is a constraint-satisfying mixed gauge/physical boundary mode; constraint histories alone need not reveal it.

This does not prove that every background or boundary formulation is unstable, and it does not test a complete nonlinear published CPBC. In particular, it is a diagnosis of the newly introduced fast radiation-boundary prototype. The old slow trumpet/production mode remains a separate unresolved issue.
