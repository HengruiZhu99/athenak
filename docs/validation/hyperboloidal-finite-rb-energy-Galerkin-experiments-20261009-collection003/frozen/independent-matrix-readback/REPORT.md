Independent finite-rb saved-matrix readback, 2026-10-09
=====================================================

The segmented-rule J=0,1,2, N=8, rb=.98 saved matrices pass the independent algebraic readback. The original J0 global-Q64 and global-Q128 matrices fail the declared weak/strong, volume and forcing checks and are retained as failures. No point kernel, assembler, generator eigensolve or evolution was run for this audit.

The verifier checks positive dense energy and Cholesky reconstruction; nodal/modal congruence; the collar principal involution, symmetrizer and projectors; independently reconstructed weak/strong and volume/boundary identities; the adjoint SAT load; independent linear solves; and measured incoming trace/sector ranks. Decorated matrices retain every original array bitwise. The sector rows match the frozen sign-plus principal classification.

| J | weak/strong scaled Frobenius error | volume identity scaled error | mixed forcing rate error in E norm | incoming rank | constraint/gauge/TT ranks |
|---|---:|---:|---:|---:|---|
| 0 | 2.09457e-14 | 1.82501e-14 | 2.02251e-11 | 4 | 2/2/0 |
| 1 | 1.16043e-13 | 9.95006e-14 | 3.40567e-11 | 8 | 4/4/0 |
| 2 | 5.72735e-14 | 5.02046e-14 | 5.49237e-11 | 10 | 4/4/2 |

All three energies are positive: minimum eigenvalues .240078, .269993 and .286646, with spectral condition numbers 55433.0, 54818.5 and 60705.0. These are energy-matrix eigenvalues, not generator spectra. The principal incoming ranks concern these finite total-J trial spaces, not an independently established local CPBC.

The unchanged thresholds are 2e-9 for mass/solve/SAT/forcing, 2e-8 for weak/strong and volume identities, 5e-11 for the point-normal adapter, and 1e12 for modal energy condition. Absolute residuals are also saved; scaled errors alone must not be read as absolute roundoff claims.

The manufactured forcing check here covers exactly one saved all-channel mixed vector per J. The independent source review verifies construction from direct point forcing, rather than defining the load as (E-Kstrong)X. The latter identity is checked afterward to explain the failed global-rule cases. Individual-channel W=1,rho,rho^2,rho^3 forcing coverage required by the held plan remains pending in a separate replay. These saved mixed-vector checks do not complete that family gate or admit growth studies.

The J0 global-Q64 and global-Q128 weak/strong errors are 1.47879e-4 and 4.44440e-6; volume errors are 1.17900e-4 and 3.51026e-6. The directly constructed source and boundary forcing match (E-Kstrong)X and -SAT X to about 1e-15 scaled, while the forced rate discrepancy matches E^-1(Kweak-Kstrong)X. The failures are preserved and the passing segmented rules use the same equations, forcing expressions and thresholds.

Independent saved-array radial comparisons (32 to 64 points per fixed segment) give maximum scaled differences 3.97112e-10, 2.96603e-10 and 2.04587e-10 for J=0,1,2. The angular 12x24 to 16x32 comparisons give 2.78187e-13, 3.23117e-13 and 2.86981e-13. All ten named matrix/load arrays were recomputed and match the owner reports. This establishes integration-rule consistency at fixed N and rb, not polynomial-degree or PDE convergence.

The J1/J2 angular comparisons use the separately reviewed BLAS contraction. AST comparison isolates the angle contraction and captured-source basename; the real flattened dot and original einsum sum the same angle/component integrand. All 23 saved J0 arrays pass the complete equivalence comparison (maximum scaled difference 5.99642e-14); B and incoming singular values are bitwise unchanged. No assembler rerun was used.

Energy-only generalized symmetric G/E maxima are 8913.72559, 8644.44940 and 8509.72994; energy-normalized full boundary trace norms are 12.39166, 12.90222 and 13.38684. The former give only finite-dimensional instantaneous energy-form bounds (half these numbers for the norm rate). They are not generator growth rates, useful uniform stability bounds, or evidence of instability. Roundoff-sized positive post-SAT flux-form maxima and their normalized skew residuals are recorded explicitly.

This energy-Galerkin/SAT problem changes bulk discretization and places an artificial finite boundary inside scri. It is not a ghost-only attribution control. The audit does not establish continuum-action equivalence by itself, exact-scri closure, complete constraint-preserving boundary data, a uniform energy estimate, finite-pulse stability or black-hole acceptance. The later single-black-hole requirement remains a wormhole-to-trumpet inner transition with the Minkowski hyperboloidal reference throughout.

All executable sources, commands, outputs, input hashes and preparation failure history in this package are frozen unchanged. The historical unnecessary SciPy import failure is retained with its correction; it occurred before scientific readback. Large external scientific matrices are retained through exact path/size/hash records, not duplicated here. Synthetic growth-helper testing has a separate package and is not a scientific propagation result.
