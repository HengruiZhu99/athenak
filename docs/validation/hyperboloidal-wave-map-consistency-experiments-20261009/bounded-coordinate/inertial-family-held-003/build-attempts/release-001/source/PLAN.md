This is a new local witness family with bounded physical inertial displacement. It does not accept any failed prior broad-coordinate run or change C0 equations, reference, physical-P/spatial-norm gauge, kappa10 damping, or input/output ordering.

For four independent radial polynomial envelopes tau,zeta,v,w in rho=x.x, prescribe physical inertial displacement/velocity Xi^T=tau, Xi^I=x^I*zeta, and dotXi^T=v, dotXi^I=x^I*w. Apply the stationary inverse embedding:

    xi^t = tau - (b/alpha)*r*zeta,
    xi^i = (Omega^2/L)*x^i*zeta,

and the same map to v,w. Here alpha is the conformal stored lapse sqrt(Omega^2+b^2), not physical alpha/Omega. The direct embedding has dR/dr=L/Omega^2 and dh/dR=b/alpha. The helper independently checks xi^t+h_i xi^i=Xi^T and partial_j(x^I/Omega)*xi^j=Xi^I, together with their velocity versions, through all available Cartesian jets. The core map is exact identity through the origin; no cross-envelope conditions or divisions by r there.

The old coordinate zeta=1 uses xi^i=x^i at scri and produces an O(1/Omega) conformal tangent. The new inertial zeta=1 uses xi^i=O(Omega^2); its time correction b*r/alpha is regular. This is a physically motivated change of witness family, not a denominator floor or a change in dynamical variables. Four radial fields remain independent. The exact outer adapter is composed from Omega=1-r^2,L=alpha=1+r^2,b=2r, so no singular reference quantities are invented.

The predeclared 629 field/point cases retain all original radii [0,.025,.049,.05,.1,.3,.45,.6,.85,.9,.95,.98,.995], three directions and four envelopes times powers0..3 plus fixed mixed. Preserve the original geometry entrywise5e-10 and final-FD2e-7 thresholds; use the fixed inherited five epsilons [1e-5,3e-6,1e-6,3e-7,1e-7]. Keep all original core/algebraic/constraint/gauge/raw22/chart20 binding gates and record absolute and scaled magnitudes. Add explicit inverse-embedding identity criterion5e-10. Record full consumed compactified tangent-jet magnitudes but make no global boundedness theorem from a finite sample.

Source preparation only: no compile, source query, operator, spectrum, evolution or boundary control before root source/recipe review. The factored lift and original physical comparator are copied byte-exact from the failed full local run; only the coordinate-family binding and additive adapter checks differ. The fixed external Omega convention is unchanged. The forthcoming exact finite-amplitude flat wave-map oracle is separate and is not assumed here.
