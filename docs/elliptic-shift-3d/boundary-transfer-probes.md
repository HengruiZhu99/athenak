# Boundary transfer audit and physical-normal Hermite atlas

The final standalone candidate is `HermiteBoundaryProbe`. It preserves original boundary values and pure-normal jets, reconstructs mixed derivatives from physical normal data, and keeps prescribed trace data independent of interior interpolation. No evolution caller, physical equation, source, boundary condition, SAT term, elliptic tolerance, or native owner changed in this audit. The frozen wormhole pole remains above its reference gate.

## Why the earlier candidates were rejected

All failures remain frozen with their sources and commands.

- A256-donor nearest global Cartesian degree5 MLS (56 basis functions) failed the value-weight L1 cap8 at11.2581. Broadening to512 donors passed geometry guards but produced a coarse N6 mixed-Hessian error5.496 versus0.26666 for the original interpolation. Using separate 2D endpoint and 3D interior fits introduced incompatible normal/angular interpolation errors.
- Using one3D fit for both interior and endpoint queries improved that error, and a physical-owner Jacobian improved donor selection. A seven-shell N6 query still failed at L1=10.2364. Selecting the same radial rings and global angular lines produced a stable Cartesian polynomial fit and passed20 geometry controls, but had a nonmonotonic seven-shell seam Hessian error at N8/N10.
- More seriously, an interior-dependent endpoint fit gives a nonzero tangential derivative when the entire boundary trace is zero. For F=(r−1)g(n), the transition Cartesian MLS gave a spurious tangential gradient around0.00424 at N6. The incoming lapse clock must not acquire tangential structure from interior values. This is a rejection of that endpoint construction, even where smooth-field errors converge.
- A physical-radius tensor interpolation with global angular MLS preserves traces and avoids inconsistent radial fits. It passed pure spherical-shell controls, including the actual seven-shell/seven-node contract. In a one-shell transition, however, constant-radius angular queries cross the central Cartesian box: some angular rays have no outer-shell data at the required radius. Its seven-node N6 one-shell case correctly failed at L1=11.764885. This is a limitation of that interpolation collar, not an incompatibility of the continuum formulation.
- The first balanced-stencil implementation used tangent-projection distance and accidentally admitted antipodal directions. Its condition guard rejected the result. Full sphere-chord distance fixed this implementation bug before further use.

The original coupled failures and the initial source/gauge remain unchanged. No failed attempt was relabeled as passing, no degree fallback was used, and no weight/condition gate was relaxed.

## Boundary jet construction

Let F be any stored global Cartesian component, including a positive-field chart coordinate where applicable. On the actual spherical outer boundary r=R, define

    f(n)=F(R,n),  G(n)=partial_r F(R,n),  H(n)=partial_r² F(R,n).

Here r=|x| is the compactified Cartesian coordinate radius. Its derivative follows Euclidean reference-radial rays; it is neither uncompactified areal radius nor the metric-unit Penrose normal. The resulting complete Cartesian jets are subsequently contracted with the evolved metric in the unchanged tensor and null-boundary kernels. No conformal-flatness assumption enters the differentiation.

For every native boundary point, compute G and H from the same original six/seven-point radial Lagrange fit, including the boundary node. Current maps have physical radius affine in the logical radial coordinate s:

    F_s = r_s G,
    F_ss = r_s² H,                     r_ss=0.

The implementation divides the native F_s and F_ss by these geometric factors before global interpolation. Patch-coordinate F_s is never treated as a patch-independent scalar. For a future nonlinear radial map, the necessary relation is F_ss=r_s²H+r_ssG; the current helper explicitly rejects a map that fails the affine-radius check rather than dropping that term.

Only the surface functions f, G and H are interpolated globally. The value interpolation uses48 globally nearest surface nodes, a query-centered gnomonic degree5 basis (21 terms), pivoted twice-reorthogonalized QR, mandatory rank and condition checks, polynomial reproduction, and a value-weight L1 cap8. All components retain the common Cartesian basis. Evolved Lambda is sampled independently; it is never replaced by a metric connection or transformed as an ordinary patch vector.

The outer surface has r_A=0 for either angular logical coordinate A. Therefore

    F_A = f_A,
    F_AB = f_AB,
    F_sA = r_s G_A + (partial_A r_s) G.

Centered fourth-order cross-face angular differences provide f_A, f_AB and G_A. The exact analytic derivative of r_s supplies the final term. For a one-layer cube-to-sphere transition,

    r_s = Delta-s [R − L sqrt(1+u²+v²)],

with the block's angular bounds included in u,v. For a spherical shell, r_s is constant in angle. These terms are essential: taking an angular derivative of a patch-coordinate radial derivative as if it were a scalar gives the wrong mixed derivative across maps.

Finally all first and second Cartesian derivative directions use the complete inverse-Jacobian chain rule,

    F_i = J[A][i] F_A,
    F_ij = H[A][i][j] F_A + J[A][i] J[B][j] F_AB.

This is the C2 boundary jet of the local extension f(n)+(r−R)G(n)+(r−R)²H(n)/2. It describes arbitrary angular and tensor-component data. It does not impose spherical symmetry, project fields, or replace interior evolution by this Taylor expression. Native normal jets retain their previous discretization and errors. Non-native G,H values are interpolation approximations; their angular derivatives are numerical derivatives of those globally interpolated physical-normal fields.

## Validation and preserved limitations

`run_hermite_boundary_probe.py --matrix all` compiles against frozen PDE headers and tests20 configurations: one, two and seven shell layers; six and seven radial reconstruction nodes; three related resolutions per shell family; and a seven-shell subdivided-face control. Maximum cells are43,000. Every surface node, a two-cell edge/corner band, and exact patch seams/corners are included. Tests are serial, run one process at a time, and have a180-second bound per process. Individual final MMS cases took under one second on this Mac; these are not scalability measurements.

The resolutions are N6/12/18 for one shell, N6/10/14 for two shells, and N6/8/10 for seven shells. Subdivided-face N6 is a topology control, not another member of those uniform-resolution sequences. The exact polynomial and nonpolynomial manufactured fields use independent analytic differentiation. Explicit additional controls are F=(r−1)g(n) and F=(r−1)²g(n). Their declared zero boundary traces are exact MMS data; no evolution state is reset.

Across all20 cases:

| Check | Measured maximum |
|---|---:|
| Whole-field constant value/first/second derivative error | exactly0 |
| Change to original native pure-normal first/second jets |2.687e-11 |
| Angular value-weight L1 |1.690 |
| Angular polynomial / radial derivative moment reproduction error |6.641e-14 |
| Tangential gradient of zero trace |2.157e-16 |
| Intrinsic surface Hessian of zero trace |6.399e-16 |
| Normal derivative of quadratic zero trace |1.809e-15 |
| Normal-tangential Hessian of quadratic zero trace |2.209e-14 |

The maximum factor-condition indicator is2.022e7, below the unchanged1e8 guard. This indicator is the product of the angular triangular QR-factor infinity condition and the scaled radial Vandermonde infinity condition. It is not a measured condition number of a full combined jet matrix. The L1 bound applies to angular value interpolation; dimensionful normal differentiation retains the original stencil and its amplification. These quantities must not be presented as a bound of8 on derivative weights.

The full and mixed derivative errors decrease across the three resolutions for each radial contract. Coarse N6 angular errors can still exceed the original owner-face result. No coarse-error gate was weakened to make those controls pass: finite/constant/trace consistency checks and resolution error measurements are distinct. The figures and JSON report every region and retain these underresolved counterexamples. The exact-query finite-difference control is a separate discretization control, not a lower bound on every error, since the Hermite method differentiates the map factors analytically.

Final MMS evidence, commands, analytic comparisons and plots:

- `evidence/scri-hermite-boundary-qualified-20261027/manifest.json`
- `evidence/scri-hermite-boundary-qualified-20261027/analysis.json`
- `evidence/scri-hermite-boundary-qualified-20261027/convergence.pdf`

## Frozen pole and complete Minkowski source checks

`run_hermite_boundary_pole.py` reads the unchanged47,088-cell AMR/log-halo wormhole midpoint checkpoint at t=.0001, with its original source tag and log chart. It checks all216 boundary points. The pole maximum changes from7.2852553e-6 (LambdaY) to9.1444520e-7 (Axx). The prescribed clock's tangential gradient is7.556e-16. The pole still **fails** the1e-7 reference gate. There is no shift solve or time advancement in this probe, and the frozen state's full regular limiting RHS is not claimed verified without a fresh consistent shift jet.

A separate complete Minkowski tensor-limit control uses the same geometry and zero stored deviations, independent Lambda=0 and analytic reference beta. Stored value/first/second jets stay exactly zero. The full tensor pole is6.662e-16, the regular tensor limit is4.441e-15, and the lapse clock rate is9.880e-16. This exercises the actual regular/pole kernels and the analytic limiting derivative; it is a local source check, not native or research evolution.

Evidence: `evidence/scri-hermite-frozen-qualified-20261027`. All original physical, lapse, flux, damping and telegrapher parameters remain unchanged. In particular no newly observed attenuated/floored spherical gauge has been imported.

## Interface needed before an optional engine experiment

The current helper is a standalone diagnostic. A production-facing implementation should split geometry from stage data:

1. Precompute native radial derivative weights, r_s and its angular derivatives, global angular interpolation plans and full J/H transforms by mesh revision, node count and scheme. Geometry plans must never retain a reference to a temporary stage-padded vector.
2. At each stage, build physical-normal component arrays G,H from current padded values and current boundary traces. Apply cached geometry plans to current f,G,H arrays. The same component-agnostic sampling must serve the22 state fields and the three algebraic delta-beta components.
3. Preserve exact native trace/normal values and constant traces. Non-native boundary-value interpolation must use surface data alone. Reuse the same surface-value contract for ghost boundary traces rather than mixing incompatible endpoint constructions.
4. Keep default sampling unchanged. Bind the optional scheme in checkpoint metadata and rebuild plans after restart/regridding. Retain original-equation shift, boundary, scri-pole, clock and constraint checks.
5. Before scientific claims, run the actual optional engine on Minkowski and the matched wormhole, check restart identity, compare all-cell/regional diagnostics, and repeat the relevant timestep/tolerance/spatial controls. A convergent standalone derivative test does not satisfy coupled3D validation.

No such engine flag or checkpoint change is included here. The native owner and shared engine remained untouched during their independent source freeze.

## Final source qualification

A dependency audit found that the first Hermite builds inherited one live project type header, `src/athena.hpp`; PDE/operator headers were frozen. Those attempts and their dependency reports are preserved. The final qualified builds copy that header and LICENSE, remove the live project-source include path, and rebuild/rerun the20-case matrix and frozen-pole/Minkowski probe in new directories. Runtime Kokkos headers and libraries are hashed before and after execution. Final dependency reports must show zero live project headers; the manifests distinguish these qualified runs from the earlier attempts. Source archives include the actual compiled project headers, not merely a current working-tree snapshot.
