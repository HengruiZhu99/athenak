# Spherical conformal telegrapher candidate

This candidate implements the user-requested relaxation mechanism in the current free-Z4c, metric-freezing elliptic-shift system. It adds a ninth evolved scalar radial component B, without constraining the eight existing stored geometry fields. B is kept separately from the geometry ABI; it is unrelated to any shift-driver variable.

Let u=alpha-A, D=partial_t-beta partial_r, and w(r) be a fixed C7 compact-support window. The equations added are

- delta D alpha = chi [w B_r + (w_r+2w/r) B];
- tau D B = kappa w u_r - B.

The first operator is chi/r^2 times the radial derivative of r^2 w B. The two angular contributions are essential: Cartesian contraction of a radial reference vector B_i=B(r)x_i/r gives B_r+2B/r. It is a flat-reference contraction, not the physical covariant divergence. Componentwise radial advection is retained. This is a disclosed conformal coordinate prescription, not a claim that the Cartesian physical gauge was transformed unchanged.

The physical conformal lapse addition is converted to a stored log-lapse rate by dividing by the total positive alpha. w and all required derivatives vanish identically in neighborhoods of both endpoints. Thus the geometric scri equations, clock, null shift and AF coefficient endpoint equations are unchanged by the direct addition. B evolves with advection and relaxation there. B starts at zero; at the AF endpoint beta=0, so its endpoint remains zero. No radial regular-center parity is imposed at the puncture. The Minkowski reference has u=B=0, for which both new terms vanish exactly. The first implementation is restricted to the full vacuum AF domain; it does not silently excise the interior.

The frozen radial lapse-trace-flux principal subsystem has speeds -beta and -beta +/- sqrt[chi(F+kappa w^2/tau)]. This sum, rather than the standalone telegrapher speed, controls the timestep guard. For c_g^2=chi F and c_T^2=chi kappa w^2/tau the frozen dispersion polynomial is s^3+s^2/tau+(c_g^2+c_T^2)k^2 s+c_g^2 k^2/tau. Positive coefficients satisfy the cubic Routh-Hurwitz criterion when c_T>0. This does not prove the nonlinear coupled elliptic/free-Z4c system stable or remove all gauge steepening.

Both characteristic families are advanced at every RK stage, using that stage's geometry and elliptic shift. The same piecewise-linear spatial interpolant supplies the ray speed and its derivative; log(J) obeys D_ray log(J)=partial_r c. Rays are injected on an exact time-lattice point and cease evolving after a domain exit. Ray positions and log(J), as well as their compensated accumulation state, are checkpointed. This eliminates coarse saved-time interpolation from the earlier diagnosis, although spatial interpolation and finite-grid regularization still require convergence checks.

The core geometry implementation remains unchanged in fast_integrator.cpp. A separate library builds telegraph_integrator.cpp around it; the Python oracle independently evaluates the flux and stored-variable conversion. All constraint diagnostics are inherited unmodified and reported for accepted states and RK stages. The original scalar and puncture experiments are not reclassified by a telegrapher success.

## AF-regular inner coupling (second candidate)

The collar version loses its smoothing mechanism as the inward front enters the taper. In the corrected N1025 run its Hamiltonian peak moves from r=.0703 at t=3 to .0391 at 3.5, .0205 at 4, and .00684 at 6. The error grows while B becomes negligible. The collar was an initial simplification, not a mathematical requirement at the AF end.

The `af-regular` variant instead sets w=1 throughout the inner region, retaining only the outer taper. No other gauge/source parameters change. At the AF end, chi=X0 r^4+O(r^5), alpha=a2 r^2+O(r^3), beta=b1 r+O(r^2), and u_r=(2a2-1)r+O(r^2). The auxiliary equation preserves B=b_B r+O(r^2), with

  dot b_B = b1 b_B + [kappa(2a2-1)-b_B]/tau.

Consequently the added conformal lapse rate is 3 X0 b_B r^4+O(r^5), and its stored logarithmic rate is (3 X0 b_B/a2)r^2+O(r^3). The leading AF lapse coefficient ODE is unchanged. This is an asymptotic derivation, not regular-center parity; the implementation continues to use one-sided puncture stencils. B(0)=0 is preserved by its own evolution equation. The additional characteristic speed is O(r^2), so it also vanishes at the endpoint. The independent Python/compiled tests exercise this variant with nonzero B and off-constraint fields. Its nonlinear behavior is still to be established.
