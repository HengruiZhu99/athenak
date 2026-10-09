# Advisory variational adjoint incoming penalty

The suggested alternative is a reasonable later finite-radius control. Evolving only X=(U,V) and recomputing the configuration derivative q from its regular solid-envelope representation keeps q=D U identically. It does not introduce independent derivative auxiliaries or lose regular origin modes. A nonlocal polynomial-space update remains in that regular trial space, provided the metric/A tangent lift and total-J channel layout are applied consistently. It changes both configuration and momentum equations at the artificial finite boundary, so it is a separately named boundary control.

Let the finite-dimensional energy be E=.5 X^T E_X X with E_X symmetric positive definite and time-independent for the fixed linear reference. It may be assembled from the overintegrated integral of (q,V)^T H(r)(q,V), plus a fixed positive U mass. Let B X=y_b=(q_b,V_b), using the complete specified normalized derivative/momentum boundary reduction. Then

    X_t|SAT=−E_X^(-1) B^T H_b k_in P+ B X
    E_t|SAT=−k_in y_b^T H_b P+ y_b=−k_in ||y+||².

Here P+ is the incoming projector in the RHS convention of FINITE-RADIUS.md; H_b P+ is symmetric positive semidefinite. Combined with the frozen principal boundary flux .5 k_in||y+||²−.5|k_out|||y−||², this gives exactly

    −.5 k_in||y+||²−.5|k_out|||y−||².

There is no momentum-only R cross term. This energy-work identity is algebraic for any such SPD E_X and matching adjoint B^T; it avoids assuming that a scalar inverse Gauss weight realizes a mixed q/V trace lift. For nonzero prescribed incoming data use the residual P+ B X−g_in in the same adjoint penalty and account for the resulting data work. Consistency is conditional on the actual continuum solution satisfying that complete boundary residual, not merely on vanishing a selected leading coefficient or on exact reference stationarity.

The added U mass should control all configurations in the kernel of the chosen derivative/normalization map. Taking a positive mass on all configuration channels is simpler than removing their constant modes. Its coefficient must have dimensions length^(-2) relative to the derivative/momentum energy, for example a recorded fixed dimensionless epsilon divided by a fixed length squared. A very small coefficient can also make E_X poorly conditioned. Its value, basis scaling, positive definiteness, solve accuracy and N/epsilon sensitivity must be measured. Overintegration with positive weights can yield a positive finite matrix without making nonpolynomial coefficient products exact.

The continuum-domain caveat is substantive. The energy controls roughly U in H1 and V in L2 at a fixed positive finite outer radius. The normal derivative and V boundary traces in B are not bounded on that energy space. For example regular L0 polynomials

    U_n(r)=(r/R)^(2n)/sqrt(n)

have integral r²|U_n'|² dr=4nR/(4n+1), bounded as n grows, but U_n'(R)=2sqrt(n)/R. Similarly V_n=(r/R)^(2n) has squared spatial norm R³/(4n+3); after normalization its boundary value grows as sqrt(4n+3)/R^(3/2). Thus the finite-dimensional trace and Riesz lift can grow with polynomial degree even though the energy is bounded. This is not a fatal objection to a wave-type boundary domain, but finite-N positivity alone is not a bounded-trace or uniform-generator theorem. The domain needs the additional regularity required to define the incoming derivative/curvature traces (such as U in H2 and V in H1) together with the intended boundary condition; a closed maximally dissipative or other uniform-limit argument remains to be established.

The principal energy identity must next be matched to the actual bulk discretization. One needs E_X J_bulk+J_bulk^T E_X to reproduce the appropriate boundary form plus explicitly bounded variable-coefficient/angular/lower-order production. The extra U mass contributes additional lower-order terms, and its control must be included. Barycentric composition, solid origin regularity and Gaussian quadrature do not establish that identity by themselves. Fixed-J angular singular-looking coefficients require regular Cartesian cancellations at the origin, and estimates may depend on J. The normal-block symmetrizer is not an all-angle, all-J three-dimensional energy certificate.

The proposed source is nonlocal through E_X^(-1), but it can still be a consistent variational finite-boundary control if the incoming residual vanishes on the intended smooth solutions and its refinement/energy properties are verified. No primitive Dirichlet data or artificial origin boundary are required by the representation. Conversely it does not supply constraint-preserving data, physical radiation data, an exact-scri manifold, a uniform r_b→S norm, or a causal interpretation identical to the Cartesian ghost scheme. Those are separate gates. No radial PDE operator or penalty solve has been formed in this assessment.
