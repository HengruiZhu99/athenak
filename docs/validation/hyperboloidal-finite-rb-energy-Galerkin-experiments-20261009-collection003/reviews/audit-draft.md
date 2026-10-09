# Actual finite-ball Z4c energy-Galerkin control

The unchanged C0 physical-P/spatial-norm linearized equations now have a concrete full-ball radial control with independently checked source, mass, energy identity and boundary forcing. J=0,1,2 at N=8 and artificial radius rb=.98 pass the recorded matrix algebra and quadrature checks. The complete fixed manufactured-forcing families contain 33, 65 and 81 fields and pass their recorded numerical checks. These are finite-dimensional operator checks; no generator spectrum, pulse propagation or stability result is supplied by this checkpoint. Production remains at 27c19d20696ea6dd4704032c51dfd026218f64f2.

The stationary reference is Minkowski with S=1,a=.5, geometric transition .05–.95 and gauge transition .45–.85. Physical-P storage/lapse, spatial-norm control xi=2, rho=1.5 and kappa1=10/alpha, kappa2=0 remain unchanged. The actual full 22-component action, complete physical metric/A lift and configuration-source spatial derivative are those in the preceding source and continuum-rate controls. No Q/C1 feedback, ghost extension or reference-independent guessed source replaces them.

Every independent solid-CG channel has its own degree-7 envelope in rho=r² over the whole ball; there is no inner boundary or r^L amplitude division. Common Gauss-Jacobi nodes and channel-dependent Jacobi modal congruences retain all 64/128/160 coefficients. The exact Cartesian core action is used below r=.05. The reduction is q=D_s U,V with complete reference-normalized configurations and momenta, including the metric contribution to the A-trace subtraction. The derivative-null configurations retain a positive U mass.

With c=alpha_hat/L, D_s=c partial_r, dV=r²/c dr dOmega and dSigma=rb²dOmega,

```
E_ij = integral (y_i^T H y_j + U_i^T U_j) dV,
K_ij = <Phi_i,L_actual Phi_j>_E,
Jbulk = E^-1 K,
H = (1-zeta) HC + zeta HD,
HC = Lleft^T Lleft,  HD = I+A1^T A1.
```

The canceled full-W left basis supplies HC. The smooth zeta blend is confined to .85–.90 where W=1, so both matrices symmetrize the same harmonic normal block. All coefficient/frame/cutoff derivatives are retained. H is a radial normal symmetrizer; this does not establish one common all-direction 3D symmetrizer.

The weak K integrates q_t by parts without differentiating momentum evolution. An independent strong integral uses the actual configuration-source derivative. Separately form R_j=y_t,j−Kn D_s y_j and Gamma=D_s(H Kn)+(div s)H Kn, then integrate

```
G_ij = integral (y_i^T H R_j + R_i^T H y_j
                - y_i^T Gamma y_j + U_i^T U_t,j + U_t,i^T U_j) dV.
K+K^T = Fboundary + Gvolume.
```

G is not reconstructed from that matrix identity. The regular core and grouped radial density define the zero origin-flux limit. Pointwise symmetry, positivity, screen rotations and coefficient derivatives have independent checks.

At rb=.98, Omega=.0396. The RHS normal incoming/outgoing coefficients are+.0004/−3.9204. The full trace B includes rb sqrt(angular weights), and the separate homogeneous incoming penalty is

```
Jsat = -E^-1 B^T HD kin Pplus B,
Pplus=(I+A1)/2.
```

Its Riesz lift updates configurations as well as momenta. Principal flux plus penalty has nonpositive boundary work. This finite-boundary principal control is not CPBC, radiative data or an exact-scri closure. The energy-Galerkin projection changes the radial bulk representation as well as the boundary, so a later comparison cannot isolate Cartesian ghosts by itself.

| J | DOFs | Coupled modal E condition | Measured incoming constraint/gauge/TT |
|---|---:|---:|---:|
|0|64|55,433|2/2/0|
|1|128|54,818|4/4/0|
|2|160|60,705|4/4/2|

All E are positive. Independent weak/strong and bulk identity discrepancies are at most 1.17e−13 scaled. Manufactured X(t)=exp(t)X0 uses pointwise forcing Phi(X0)−L_actual Phi(X0) and nonzero incoming data Pplus B X0; every independent channel with W=1, rho, rho², rho³ plus the all-channel mixed field passes. Maximum coefficient errors are 7.367e−11, 9.248e−11 and 1.021e−10 for J=0,1,2; maximum scaled E-norm errors are 7.651e−11, 1.917e−10 and 1.403e−10, below the fixed 2e−9 threshold. The forcing is not defined by subtracting the assembled operator. Shared-source forcing tests assembly; independent core/source-derivative/continuum-subsidiary gates check the source separately. Exact polynomial masses and modal congruences pass without diagonal mass substitution.

The original global rho rules Q64/Q128 failed weak/strong, bulk and forced-coefficient gates. Their records remain unchanged. One named refinement splits quadrature only at fixed cutoff endpoints 0, .05, .45, .85, .90, .95, rb. The paired 32/64-node panel rules leave every equation, trial function, source, SAT coefficient and tolerance unchanged. The rule with 32 nodes per panel still fails the stricter forced-coefficient gate; 64 nodes per panel pass, with the later replay covering the complete fixed forcing family. The original per-rule reports tested the mixed field only. Paired E/K/G/J/SAT/forcing changes are below 4e−10, and 12x24 versus 16x32 angular changes below 3.3e−13. These numerical comparisons do not prove quadrature exactness or degree-uniform estimates.

A contraction-only BLAS implementation changes the angular bilinear sum into a flattened dot product. Full saved-array readback checks all 23 arrays, keys, shapes, dtypes and finite values against the original contraction; maximum scaled difference is 5.997e−14, with unchanged trace/rank data. Old unoptimized runs and failures remain preserved. The optimization changes neither the PDE nor the projection or penalty.

The finite generalized G/E maxima are approximately 8913.73, 8644.45, 8509.73; full trace norms are 12.39, 12.90, 13.39. These are finite-N energy/trace diagnostics, not generator spectra or useful uniform stability bounds. A separate analytic point diagnostic now measures constraint production by the N8/J0 projection and SAT for 14 witnesses at 21 locations. Both preceding ordinary-FD attempts remain failed: one has resolved fourth-order truncation above its last-increment threshold near the outer boundary, and the finer-step attempt encounters interior roundoff. The complete nongauge continuum comparator C_ref[L_actual Phi(X)] remains unresolved, so the original full projection-defect gate remains held. N12/N16, the separate rb=.995 problem, finite-matrix eigenanalysis and guarded linear propagation remain later checks. J<=2 also omits J3 content of the original vector pulse and nonlinear angular mixing. The substantial angular Minkowski gauge pulse and later single-BH wormhole-to-trumpet transition remain unvalidated; both must retain the Minkowski hyperboloidal reference throughout.
