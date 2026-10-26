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
