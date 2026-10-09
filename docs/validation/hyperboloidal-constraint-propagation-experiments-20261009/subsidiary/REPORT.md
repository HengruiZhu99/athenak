# Continuum subsidiary audit: actual C_Z4c=0 tensor kernel

Restoring gradients of the background/operator coefficients restores pure
gauge constraint tangency. A nonzero constraint residue on a primitive frozen
Fourier eigenvector therefore does not classify a physical subsidiary eigenmode.
This is independent of the value-only lapse/shift control: H/M/Theta/Z do not
depend on gauge fields, so their instantaneous propagation cannot contain the
gauge RHS. Actual geometric dependence on lapse/shift and their jets is retained.

The dual-number derivative of the unchanged ConformalRHS and EvolvedConstraints
is exact to floating-point arithmetic. All20 independent fields satisfy det=1
and tracefree reconstruction including spatial jets. The complex plane phase
is differentiated analytically. Fourth-order differences differentiate only
smooth operator coefficients, retaining the nonflat reference gradients.
S=1,a=.5,geometry .05-.95; radii .3,.5,.75,.85,.9,.95,.98; radial/oblique k=
0,1,2,4,8,16,32,64,128,256; h=.002,.001,.0005,.00025,.000125; kappa_input5/10.
Each damping has700 full20 chain-rule comparisons. The finest matrix-relative
closure error is at most3.6034e-9. Static constraint gauge columns are exactly0.

At radial k64 the largest pure gauge Cdot residuals are:

| radius | naive frozen QL | h=.002 with coefficient gradients | h=.0005 | h=.000125 |
| --- | ---: | ---: | ---: | ---: |
| .75 | 7034.93 | 2.27842e-4 | 1.01171e-6 | 2.59728e-6 |
| .85 | 3738.10 | 1.73609e-3 | 6.79423e-6 | 1.10410e-6 |
| .95 | 5194.58 | 2.21610e-2 | 8.59230e-5 | 3.95513e-7 |
| .98 | 5358.23 | 1.29892 | 4.83737e-3 | 1.89405e-5 |

The errors decrease at fourth order until spatial second-difference roundoff
limits the interior cases. At .98 the last refinements still have ratios16.
The direct symbolic/compiled Theta and Z identities and full physical ADM
H/M closure with explicit S_ij are in DERIVATION.md and subsidiary.hpp.
The actual C0 isotropic addition B contains both spatial-Z terms and a
Theta term; these are retained, not replaced by C1 equations.

The derived eight-constraint local generators freeze the constraint fields,
retaining coefficient gradients in differentiated S. All140 sampled kappa10
generators through k256 have negative roots (worst Re=-1.10273 at r.3,k2oblique).
At kappa5 the outer scalar branch grows: r.98,k256 lambda=32.43913+59.79164i.
At r.98,k16384 its phase/k=.00046538 approaches beta^n+light=.0004, the nearly
stationary incoming branch; Re=47.86925. The matching kappa10 root has
Re=-15.74893, phase/k=.00047584. The high-k balancing is a matrix similarity
for numerical eigenvalue conditioning, not a runtime field weight/falloff.
Eigen backward residuals in the full gate are below1e-12. These are local
subsidiary generators, not global continuum eigenvalues or energy bounds.

An exact SymPy calculation reduces the principal constraint system to four
waves with positive Fourier energy in U=H+2divZ and V=M+gradTheta for |k|>0.
The lower-order damping does not establish a uniform scri estimate. At outer
S1,a.5,kappa10 the isotropic source contributes +8r*Theta/Omega² to radial Mdot;
ordinary M²+Theta² damping is only O(1/Omega). An adapted energy/cross-term
calculation or justified regularity/Hardy control is needed. ENERGY.md states
the principal energy and the precise limitation. No arbitrary Theta falloff,
nonlinear regularity closure or C_Z4c=1 implementation is introduced.

The separate native global C_h L_h gauge tangent is nonzero, while continuum
Cdot for pure gauge data vanishes. This identifies a numerical Bianchi/product-
rule defect for the native projector/boundary/global audit to quantify. It does
not identify its exact source or prove that it alone causes native growth.
The native unweighted H/M/Z RMS is not the principal constraint-wave energy.

Provenance: checked receipt launch HEAD b37a20f2, unchanged runtime implementation
27c19d20, exact compiler commands/header hashes and all large output hashes in
receipt.json. check_subsidiary.py records identity/eigen checks in
subsidiary-report.json; check_principal_energy.py proves the exact principal
wave identity. Its original structural matrix-equality assertion failed before
semantic simplification; complete failing source/command/stderr are retained
under failed-principal-structural-assertion/. An initial C++ int*complex literal
compile typo is separately documented; it did not produce an accepted audit.
The preliminary tangent.json/tangent-subsidiary.json outputs are superseded;
only tangent-kappa5/10.json and generator-kappa5/10.json are final receipt outputs.
No production source, runtime option, default, boundary or native build changed.
