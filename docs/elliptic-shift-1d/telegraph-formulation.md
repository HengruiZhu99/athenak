# Spherical conformal telegrapher candidate

This candidate implements the user-requested relaxation mechanism in the current free-Z4c, metric-freezing elliptic-shift system. It adds a ninth evolved scalar radial component B, without constraining the eight existing stored geometry fields. B is kept separately from the geometry ABI; it is unrelated to any shift-driver variable.

Let u=alpha-A, D=partial_t-beta partial_r, and w(r) be a fixed C7 compact-support window. The equations added are

- delta D alpha = chi [w B_r + (w_r+2w/r) B];
- tau D B = kappa w u_r - B.

The first operator is chi/r^2 times the radial derivative of r^2 w B. The two angular contributions are essential: Cartesian contraction of a radial reference vector B_i=B(r)x_i/r gives B_r+2B/r. It is a flat-reference contraction, not the physical covariant divergence. Componentwise radial advection is retained. This is a disclosed conformal coordinate prescription, not a claim that the Cartesian physical gauge was transformed unchanged.

The physical conformal lapse addition is converted to a stored log-lapse rate by dividing by the total positive alpha. w and all required derivatives vanish identically in neighborhoods of both endpoints. Thus the geometric scri equations, clock, null shift and AF coefficient endpoint equations are unchanged by the direct addition. B evolves with advection and relaxation there. B starts at zero; at the AF endpoint beta=0, so its endpoint remains zero. No radial regular-center parity is imposed at the puncture. The Minkowski reference has u=B=0, for which both new terms vanish exactly. The first implementation is restricted to the full vacuum AF domain; it does not silently excise the interior.

The frozen radial lapse-trace-flux principal subsystem has speeds -beta and -beta +/- sqrt[chi(F+kappa w^2/tau)]. This sum, rather than the standalone telegrapher speed, controls the timestep guard. For c_g^2=chi F and c_T^2=chi kappa w^2/tau the frozen dispersion polynomial is s^3+s^2/tau+(c_g^2+c_T^2)k^2 s+c_g^2 k^2/tau. Positive coefficients satisfy the cubic Routh-Hurwitz criterion when c_T>0. This does not prove the nonlinear coupled elliptic/free-Z4c system stable or remove all gauge steepening.

Both characteristic families are advanced at every RK stage, using that stage's geometry and elliptic shift. The same piecewise-linear spatial interpolant supplies the ray speed and its derivative; log(J) obeys D_ray log(J)=partial_r c. Rays are injected on an exact time-lattice point and cease evolving after a domain exit. Ray positions and log(J), as well as their compensated accumulation state, are checkpointed. This eliminates coarse saved-time interpolation from the earlier diagnosis, although spatial interpolation and finite-grid regularization still require convergence checks.

The initial uniform-grid geometry implementation remained unchanged in fast_integrator.cpp. A separate library builds telegraph_integrator.cpp around it; the Python oracle independently evaluates the flux and stored-variable conversion. All constraint diagnostics are inherited unmodified and reported for accepted states and RK stages. The original scalar and puncture experiments are not reclassified by a telegrapher success.

## AF-regular inner coupling (second candidate)

The collar version loses its smoothing mechanism as the inward front enters the taper. In the corrected N1025 run its Hamiltonian peak moves from r=.0703 at t=3 to .0391 at 3.5, .0205 at 4, and .00684 at 6. The error grows while B becomes negligible. The collar was an initial simplification, not a mathematical requirement at the AF end.

The `af-regular` variant instead sets w=1 throughout the inner region, retaining only the outer taper. No other gauge/source parameters change. At the AF end, chi=X0 r^4+O(r^5), alpha=a2 r^2+O(r^3), beta=b1 r+O(r^2), and u_r=(2a2-1)r+O(r^2). The auxiliary equation preserves B=b_B r+O(r^2), with

  dot b_B = b1 b_B + [kappa(2a2-1)-b_B]/tau.

Consequently the added conformal lapse rate is 3 X0 b_B r^4+O(r^5), and its stored logarithmic rate is (3 X0 b_B/a2)r^2+O(r^3). The leading AF lapse coefficient ODE is unchanged. This is an asymptotic derivation, not regular-center parity; the implementation continues to use one-sided puncture stencils. B(0)=0 is preserved by its own evolution equation. The additional characteristic speed is O(r^2), so it also vanishes at the endpoint. The independent Python/compiled tests exercise this variant with nonzero B and off-constraint fields. Its nonlinear behavior is still to be established.


## Fixed numerical maps for resolving the inward layer

The uniform-grid AF-regular run still brings the late front into the first physical cell. A fixed numerical map distributes collocation points while retaining r as the coordinate in all continuum equations. It is neither inner excision nor a change of the Einstein constraints. Two declared controls are used: r=sinh(s*x)/sinh(s), and an inner-only map r=x[a+(1-a)S(x/.4)] with a=s/sinh(s) and a C7 Bernstein smoothstep S. The second map has r=x and J=dr/dx=1 exactly for x>=.4. The strength s is fixed under refinement; both include the AF endpoint and actual scri.

Derivative and quadrature weights are constructed directly on the physical r nodes. The seven-point physical-r polynomial stencils and elliptic integration therefore retain their original continuum targets. The map changes numerical resolution, not tensor transformation rules. The step is dt=CFL*dx. Stage guards use max(abs(c)/J), dt/tau, and the KO bound dt*epsilon/(dx*min(J))<=2. The last bound is conservative relative to RK4's negative-real-axis stability limit.

On the mapped grid, the dissipative term is -epsilon J^{-1} D3^T W D3/(64 dx), with the same excluded AF coefficient column and physical-r taper as before. Thus sum_i J_i u_i Q_i=-epsilon sum_i W_i(D3 u)_i^2/(64 dx)<=0. It vanishes under refinement at fixed map. The geometry and auxiliary field use the same construction. Boundary penalties use the local scri spacing dx*J(scri). Full-domain radial RMS diagnostics use r^2 J weights. No nodes are removed from constraints or field comparisons.

The optional numerical-map setter leaves the geometry ABI and uniform path intact; a saved pre-map library gives bitwise identical uniform states and diagnostics for the checked nonzero-flux evolution. The full 52-test suite passed after the geometry-kernel additions. Independent mapped checks include physical-coordinate polynomial quadrature, smooth-function derivative convergence, the weighted KO identity, nonzero-flux Python/compiled rates and RK stages, characteristic interval selection, and exact checkpoint restart.

The global sinh pilot improves the late inner error but coarsens scri. The inner-only map is a separate controlled response to that failure. Neither mapping is accepted on a coarse run alone.


## What characteristic compression does and does not show

For the original gauge the frozen lapse-trace speeds are c_±=-beta±sqrt(chi*F), F=alpha^2+4 nu Omega^2 alpha. Both the metric coefficient and the shift are essential. Characteristics are integrated as dot r=c_± and dot log(J)=partial_r c_± using each RK stage. At t=2.5 the zero-flux controls give minimum inward J=.000652882 (N513) and .0000126791 (N1025); the kappa=.001 candidate gives about .142 on the resolved grids. At t=3 the baseline minima shrink further to 1.26247e-5 and 1.21675e-8. These stage results supersede the much weaker compression inferred from snapshots separated by .1 in time.

This trend supports a nonlinear steepening diagnosis, but no characteristic intersection in a regularized finite-grid velocity field is a proof of continuum noncrossing. Conversely, tiny J is not an observed intersection. The source system also has elliptic shift feedback, radial geometry and conformal sources. The simple Bona–Masso condition 1-f-alpha*f'/2=0 applies to its specified one-dimensional model, not automatically to this full system; see [Alcubierre, Eqs. (21)–(24)](https://arxiv.org/html/gr-qc/0503030). For the local original coefficient f=F/alpha^2, the corresponding expression is -2 nu Omega^2/alpha, which is nonzero in the interior. This is a useful diagnostic comparison, not a shock theorem for our formulation.

The telegrapher extension changes the principal speeds and damps the frozen three-field subsystem. Its damping does not guarantee nonlinear shock avoidance. Late AF coordinate compression remains a separate resolution challenge even after the initial steep front is smoothed.


## Dissipation for stronger fixed maps

The optional `mapped_ko=local` discretization replaces the unit coefficient inside the mapped KO quadratic form by a harmonic mean of the four nodal Jacobians:

  Jbar_i = 4 / (1/J_i + 1/J_(i+1) + 1/J_(i+2) + 1/J_(i+3)),
  Q = -epsilon/(64 dx) J^{-1} D3^T diag(W Jbar) D3.

This is a numerical dissipation change, not a modification of the continuum equations. It equals the original operator on a uniform grid. All geometric and auxiliary fields use the same operator; the AF coefficient column remains excluded. The old mapped mode, `physical`, retains Jbar=1 and remains available as a control.

For positive J and 0<=W<=1, the dr energy identity is

  u^T J Q u = -epsilon/(64 dx) sum_i W_i Jbar_i (D3 u)_i^2 <= 0.

Moreover Jbar_i<=4 J_j for every j in its four-point stencil. Applying the weighted Cauchy inequality to the third-difference coefficients, whose absolute values sum to 8, and summing the stencils gives

  -u^T J Q u <= (4 epsilon/dx) u^T J u.

Thus the dissipative spectrum lies in [-4 epsilon/dx,0] in this inner product, independently of min(J). The conservative guard dt*epsilon/dx<=.5 places this operator within the RK4 negative-real-axis interval. It is combined with the stage characteristic and relaxation guards; it is not a stability theorem for their sum or for the nonlinear Einstein system. For a fixed smooth positive map, the smooth harmonic weights preserve a vanishing fifth-order interior KO truncation term. The boundary stencil remains the explicitly documented puncture stencil.

The independent tests verify the full similarity-transformed dissipation matrix, its weighted energy identity and spectral bound, as well as nonzero-flux Python/compiled rates, RK stages, rays, and restart on strongly clustered maps. These checks justify testing stronger maps without an artificial KO timestep restriction proportional to the smallest physical cell. Nonlinear convergence is still an experimental question.


## Exponential inner resolution control

The `inner-exp` map uses E(x)=.5 expm1(2 b x)/expm1(b), with fixed b=16 in the declared three-grid experiment. It equals E for x<=.3, blends as r=(1-S)E+S x through .3<x<.5 with the same C7 smoothstep, and equals x beyond .5. On the blending interval E<=x and S'>=0, so J=(1-S)E'+S+S'(x-E)>0. In the exponential region away from the first cell, dr/r is approximately 2 b dx. Both the map and its derivative are computed directly at each x, so nested grids share identical physical nodes. The map remains a discretization choice; all physical-r tensor equations and auxiliary coupling are unchanged.

This control addresses an observed weakness of the stronger `inner-c7` maps: reducing their first cell can move the front into a poorly resolved transition at larger r. The exponential map is not accepted from its construction; its own transition and all boundary/constraint errors are included in the tests.

## Higher-order numerical controls

The optional `--width 9` changes the centered/one-sided physical-r first and second derivatives and the biased advection stencil from seven to nine nodes. Elliptic quadrature remains independently fixed by `--quadrature-width`; choosing a wider derivative stencil does not silently change it. The continuum equations, AF coefficient evolution, and scri limit equations are unchanged. Wider stencils can be poorly conditioned on a very coarse strongly mapped grid; the observed N33/b16 failure and its CFL rejection are preserved.

`--ko-order 8` uses a fourth forward difference with coefficients (1,-4,6,-4,1), in place of the original third forward difference. For p=3 or 4, write

  Q_(2p) = -epsilon/(2^(2p) dx) J^-1 Dp^T diag(W Jbar) Dp,
  Jbar_i = (p+1) / sum_(j=0)^p (1/J_(i+j)).

The physical-KO option retains Jbar=1. The local-KO option satisfies the exact dr-energy identity

  u^T J Q_(2p) u = -epsilon/(2^(2p) dx) sum_i W_i Jbar_i (Dp u)_i^2 <= 0.

Because Jbar_i<=(p+1)J_j for every stencil node and the absolute difference coefficients sum to 2^p, the magnitude of the dissipative spectrum is at most (p+1)epsilon/dx. The local-KO timestep guard is dt*epsilon/dx<=2/(p+1); the characteristic and relaxation guards also remain in force. This controls the isolated dissipative operator, not the full nonlinear coupled system.

On a fixed smooth map, Q_(2p) has an interior truncation term of order dx^(2p-1). Thus KO6 and KO8 denote sixth/eighth derivative dissipation with fifth/seventh-order interior truncation respectively. Both exclude the AF coefficient column and taper to zero before scri. Geometry and auxiliary flux use the same order. Neither changes the Einstein constraints or projects their errors. The default width7/KO6 path retains the checked bitwise baseline behavior.

The independent implementation checks cover the weighted energy identity and complete dissipative spectrum for both orders, nonzero-flux Python/compiled rates, free-constraint RK stages and rays, and exact restart. High-width b16 rate agreement is tested at N129; N33/65 transition stencils are too ill-conditioned for the original absolute rate tolerance. The underresolved N33 evolution is a negative CFL-guard test, not a claimed solution. The four-case pilot varies differentiation width and KO order separately before any combined candidate is selected.

## Outer-resolution control with the original derivative stencil

`inner-exp-scri` retains the inner-exp map exactly for x<=.8. Beyond that point let L=1-.8 and y=(x-.8)/L, and use

  r=x+.5 L y^8(1-y),
  J=1+.5 y^7(8-9y).

This is C7 at x=.8, fixes r(1)=1, and gives J(1)=.5. It is monotone: .5<=J<=1+.5(7/9)^7 in this outer interval, and 1-r=L(1-y)(1-.5y^8)>=0. The inner front resolution is unchanged; slight coarsening earlier in the outer interval compensates for finer scri spacing. This changes the numerical node distribution only. All physical-r derivative/quadrature weights, mapped dissipation, stage-speed guards and local scri SAT spacing use the actual map Jacobian.

The control retains width7 after the observed width9 scri failure. It addresses the early pole-residual maximum, reached around t=.2-.25, separately from the late inner Hamiltonian error. A prepared pilot compares both maps at N513 through t6 and N1025/2049 through .5. The short fine runs cannot establish late boundary stability or the full objective; any selected map must subsequently pass complete main/extension convergence and independent controls.

## Optional C1 characteristic interpolation

`--ray-interpolation pchip` replaces only the diagnostic velocity interpolation. Each RK stage constructs monotonic cubic Hermite interpolants for both sampled coupled speeds in physical r, with Fritsch–Butland weighted harmonic nodal slopes and one-sided limited endpoint slopes. The ray position uses the interpolated speed and log(J) uses the analytic derivative of that same cubic. This is C1 across radial nodes; its second derivative need not be continuous. The default remains the original linear interpolant. See the [SciPy algorithm description](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

This choice changes neither the geometry/flux RHS nor their dissipation, elliptic shift, stage guards, state layout or compensated updates. The mode is recorded in the run configuration and checked on restart. Exits remain terminal. Geometry, flux and their compensation are verified bitwise against the linear mode; the default linear trajectory is also checked against the preceding frozen library. An optional isolated SciPy 1.13.1 installation supplies an independent value/derivative and actual nonzero-flux RK-stage oracle. No SciPy dependency is added to the solver.

For any Lipschitz interpolated speed, the finite-grid ray flow has J=exp(integral partial_r(c) dt)>0 while it remains regular. Thus a positive tracked J cannot by itself establish continuum absence of crossing. The relevant evidence is convergence of the speed gradients, ray compression and geometry under both spatial and independent timestep refinement. Improving diagnostic smoothness must not be presented as smoothing the evolved lapse.
