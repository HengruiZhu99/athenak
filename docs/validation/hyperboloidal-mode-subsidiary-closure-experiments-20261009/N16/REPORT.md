# Saved C0 approximate-mode constraint intertwining audit

The saved N16 spatial-norm C0 approximate mode has a substantial defect between its actual native constraint evolution and a separately discretized coefficient-aware continuum subsidiary system, including on the 32 points whose full nested native stencils avoid all ghost reads. This establishes a sampled finite-grid intertwining failure. It does not identify a unique origin of the growing mode, an autonomous unstable interior subsystem, or a continuum instability.

No new evolution, eigensolve, native long run, matrix generation, or production edit was performed. The original approximate/pseudospectral mode, generator, native callback, and all previous evidence are read-only inputs.

## Actual operators and normalization

The original matrix is the continuous projected20 C0 spatial-norm generator on N16/span2.2, 1640 active Cartesian sphere points, h=.1375, Omega_min=.0026953124999994555, S=1,a=.5, reference transition(.05,.95), symmetric quadratic primitive continuation, kappa_input=10/kappa2=0, native Dx/Dxx/Dxy, Lx, and KO=.1. This is distinct from the final-stage-only native finite-RK3 map.

The pinned unit-free20 complex candidate has approximate lambda=1.916338996568537+6.790995420746044i and actual cached generator residual5.886773215552094e-7. It remains approximate/pseudospectral: there is no eigenvalue error bound, rightmost-spectrum claim, or continuum mode identification.

Write q=C_h v and r=C_h(J20 v). The actual cached J20v is used, rather than lambda*v. The original native callback constructs analytic full reference jets plus the native differentiated and continued primitive perturbations, then evaluates the centered directional derivative of EvolvedConstraints. Both real and imaginary signed covectors are preserved before taking norms. The order is physical H, physical M_x/M_y/M_z, physical Z_x/Z_y/Z_z, physical Theta. Stored Theta is already physical; neither M nor Theta is divided by Omega. Theta is appended using the original pinned 22-by20 lift.

The comparison K_c q applies the byte-identical frozen reference-only C0 Subsidiary() with all analytic background coefficient gradients, discretizing constraint jets with the actual native centered Dx4, independent Dxx4, and mixed Dxy. Thus K_c is a separately discretized continuum identity about the analytic stationary Einstein reference, not an identity that C_h and J20 must satisfy at finite h.

H/Theta RMS is unweighted nodal complex RMS. M/Z RMS contracts their covectors with the analytic Penrose inverse chi*gtilde_inverse before averaging. These are component norms, not a physical energy bound. Projected complex rates are <q_c,r_c>/<q_c,q_c> in that same group inner product; they are not subsidiary eigenvalues.

## Exact stencil coverage

S2 includes the center, axis offsets±1/±2, and all two-axis mixed±1/±2 corners. S3 adds axis±3, matching all actual Lx input reads; KO uses only native admitted active four-point lines. The full nested mask checks every S2+S3 offset directly against the active sphere. No radial collar approximation is used.

| Mask | Points | Radius range | Fraction of mode H²/M²/Z²/Theta² |
|---|---:|---|---|
| Centered constraint stencil active |432|.11907849–.62634231|.592341/.0639981/.00299711/.149668|
| Centered plus Lx constraint stencil active |408|.11907849–.62634231|.592319/.0619731/.00277558/.148886|
| Full nested primitive and constraint stencil active |32|.11907849–.22801795|.163710/.00795966/.000135640/.0791990|

The strict32 sample has almost no leverage over the mode's outer M/Z support. Its nonzero defect demonstrates a local discrete consistency failure for this sampled mode, but the mode itself is a global vector whose interior shape can depend on the outer closure.

## Strict32 result

All rows below use the identical full nested32 mask and signed complex fields. Values list H/M/Z/Theta.

| Quantity | RMS or relative RMS |
|---|---|
| Actual r=C_hJ20v |.390295345/.0979774251/.00477680191/.00725605053|
| Centered K_cq |.405069701/.100381815/.00663614643/.00378769254|
| r−K_cq |.243861384/.0836103890/.00767921077/.0104858865|
| (r−K_cq)/r |.624812/.853364/1.607605/1.445123|
| [r−(K_c+U_c+Q_c)q]/r |.670352/.870750/1.602048/1.450405|

Actual group projected rates reproduce the saved approximate lambda to a few parts in1e-6. The centered subsidiary group rates are −2.00843+6.75018i,1.58619+4.39669i,7.78750−.930589i,−1.99969−2.42667i. These contractions quantify mismatch on the same vector; they are not an eigenanalysis of K_c.

## Separate transport/dissipation commutators

U is the native primitive Lx−beta*Dx correction, Q is native primitive KO=.1, each followed by the original algebraic Restrict. The constraint-side U_c uses the identical Lx−beta*Dx on each physical constraint component; Q_c uses the identical active-line KO on each constraint component. No commutation through C_h or through background coefficients is assumed.

The receipt separately reports C_h(J−U−Q)v−K_cq, C_hUv−U_cq, and C_hQv−Q_cq. The strict32 absolute commutator RMS values are:

| Difference | H | M | Z | Theta |
|---|---:|---:|---:|---:|
| Centered primitive minus centered subsidiary |.281530630|.0812684511|.00737307735|.0105242121|
| Upwind commutator |.0427752500|.00595915190|.000757478693|0|
| KO commutator |.0116985176|.00284001642|.000480666411|0|

Matching the constraint-side Lx/KO therefore leaves a large strict-interior defect. The measured defect includes finite-h Hessian/product-rule/diagnostic/projector consistency effects; this single vector does not separate them uniquely.

## Chosen same-ray constraint extension

To inspect the excluded outer support, a separate comparator applies the original symmetric quadratic primitive ghost plan independently to the eight constraint components. All63576 donor references are strictly active and nonrecursive; weight-sum error≤1.3323e-15. Constants and quadratics are preserved to roundoff in the manufactured tests. This is a chosen extension of nodal constraints, not the constraint extension induced by continuing the primitive fields and then forming C_h.

Strict and same-ray callbacks agree bitwise on all mutually admitted centered/Lx/KO rows. With the chosen extension, all-grid centered-defect RMS is1.19507705/167.778903/.260984678/.0215438618, versus actual C_hJ20v RMS.134743832/.153402304/.0572924162/.00360158443. The M defect ratio1093.718 is an extension-dependent comparator mismatch; it is not evidence that the native primitive ghost closure is unstable or wrong. In the outer r≥.95 bin (264 points), the chosen centered M defect RMS is418.131569, peaking at r=.998651434. Full radial counts, RMS, peaks, complex rates, and both denominator normalizations are in results.json.

## Numerical checks, scrutiny, and provenance

The original native RHS action independently agrees with the cached Jv to6.7474e-8 in generator state units at max-component perturbation1e-4. Applying C_h to that actual RHS action changes r by at most9.22e-8 relative by component group; the smaller1e-5-scale constraint callbacks agree across four amplitudes to≤6.10e-9. Constraint-side K_c amplitude sensitivity is≤5.07e-7, far below the reported mismatch. Native r−lambda*q is≤1.52e-7 relative, so the approximate-mode error does not explain the O(1) strict defect. The C_h linearity readback RMS is≤1.03e-9.

Constant/quadratic analytical-jet Subsidiary oracles pass on strict and chosen-extended support with maximum scale-normalized error3.65e-16; native upwind correction and KO annihilate these data to roundoff. This checks the derivative bridge and unchanged kernel binding, not a discrete closure theorem.

Continuum agent read-only scrutiny pinned comparator.cpp SHA0a8b8589d295496b22f24cf16731d28b0e1ab15f90a0216cdecbe73966a642f8 and found the frozen Subsidiary binding, physical normalization, exact nested support, and separately projected native U/KO appropriate. Its scope qualifications are preserved verbatim in source-review.json.

The current callback was compiled with AppleClang arm64 C++17/O3/DNDEBUG and pinned Kokkos static libraries; build-provenance.json captures the exact command, compiler identity, every compiler dependency, archive hashes, and executable5b3be9505265f30c3586dacf59000764195cfa3ed9fa31077d5f39d2efe80033. All newly compiled production headers match27c19d20. The final diagnostic took approximately2.3s after server startup. Earlier scratch failures (nonassignable field adapter, then copying an already-read-only pinned input) are retained; neither produced scientific data. The first completed diagnostic is preserved separately because additional constraint-operator amplitude sensitivity was then justified by the outer coefficients.

This audit supports investigating discrete compatibility and boundary-fitted full-tensor controls. It neither clears the outer closure nor proves it causes the saved global growth. No continuum subsidiary classification, physical energy bound, eigenvalue certification, or whole-system stability claim follows.
