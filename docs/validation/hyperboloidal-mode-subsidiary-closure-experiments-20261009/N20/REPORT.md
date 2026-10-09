# Existing default-N20 mode: subsidiary comparator

The separately discretized coefficient-aware C0 subsidiary identity still has substantial mismatch against actual native C_hJ20v on strictly active nested stencils at N20. In the same small physical ball as the N16 audit, H mismatch is smaller while M/Z mismatch is larger. These are two different approximate modes on grids with different closest-shell phase; the result does not establish an order, continuum instability, or unique bulk/boundary cause.

This fresh directory preserves the original N16 freeze, all N20 histories/operators, and all previous mode-search outputs. No propagation, new matrix generation, global eigensolve, shift-invert, or native evolution was run. Parent explicitly authorized the limited warning-free replay of the already specified32-by32 and16-by16 reduced Ritz exports, because the original stored vector keys referred to maximum ranks below the history-noise screening threshold.

## Pinned mode and actual operator

The default N20/span2.2 C0 spatial-norm operator has3112 active points, h=.11, Omega_min=.022925, S=1,a=.5, reference(.05,.95), kappa_input10/kappa2zero, symmetric quadratic primitive ghosts and native KO.1. The prior source/actual22/final-only step gate is read-only, pinned by immutable-C0-N20-default-span-v2-20261009/index.json8ec4c4dd.... This audit applies its continuous projected20 generator; it makes no new native finite-RK claim.

The warning-free export replays exactly the prior t2–6 gauge window, stride2, rank32 Ritz calculation, with explicit dense contractions. It reproduces lambda=1.9182802861785322+6.805508834100131i exactly and actual J20 residual3.823879262878925e-7. The singular fraction is1.4100894213604591e-9, above the heuristic1e-10 threshold. An independent fixed t4–6, stride1, rank16 replay gives a phase-aligned vector distance6.460480773884449e-8 and reproduces its prior lambda/residual. RuntimeWarnings were promoted to errors; neither replay emitted one. They remain approximate/pseudospectral vectors without eigenvalue error bounds or global-rightmost certification.

The new callback differs from the scrutinized N16 source only in the grid constructor: n26,h2.2/20,first−.5*25h replaces n22,h2.2/16,first−.5*21h. The exact diff, source generator and pins are retained. All frozen Subsidiary/native derivative/primitive correction/lift code is unchanged.

As in N16, q=C_hv and r=C_h(J20v) use the actual cached J20 action, with independent actual native RHS verification. Signed complex physical H, M covector, Z covector, and stored physical Theta are preserved; no Omega rescaling is applied. M/Z norms contract with the analytic Penrose inverse chi*gtilde_inverse. These component norms and projected complex rates are not physical energy bounds or subsidiary eigenvalues.

## Exact masks and coverage

| Mask | Points | Radius range | Mode H²/M²/Z²/Theta² fraction |
|---|---:|---|---|
| Centered constraint stencil active |1088|.09526279–.71921833|.737504/.123529/.00739630/.178049|
| Centered plus Lx active |1064|.09526279–.68474448|.730369/.122412/.00718326/.177628|
| Full nested primitive/constraint stencil active |184|.09526279–.39277856|.313221/.0304315/.00143658/.104521|

Masks explicitly enumerate S2 centered axis/mixed offsets and S2+S3 nested primitive support, where S3 adds native Lx axis±3. No radial collar estimate is used. Even the enlarged nested mask covers only3.04% of M² and.144% of Z², so it cannot clear the outer closure.

On all184 full-nested points, actual r RMS H/M/Z/Theta is.247266246/.0709465939/.00487520021/.00343411829. Centered K_cq RMS is.209044297/.170676270/.00732757494/.00581519941, and defect RMS is.0954305990/.140112029/.00483101571/.00789911818. The scale-invariant defect/actual ratios are.385943/1.974894/.990937/2.300188; matching constraint-side native Lx and KO gives.498149/2.046781/1.082135/2.360349. Full signed-group contractions, absolute peaks/radii, radial bins and the separate C_hU−U_cC_h and C_hQ−Q_cC_h commutators are retained in results.json.

## Same physical ball

Both grids have32 points inside r≤.22801795433693378, the N16 full-nested maximum radius. Each point in this ball passes its own full-nested mask. The modes are not equated or rescaled to the same Euclidean dimension; ratios are independently scale invariant.

| Grid | Actual C_hJv RMS H/M/Z/Theta | Centered defect RMS | Centered defect/actual | Matched Lx+KO defect/actual |
|---|---|---|---|---|
|N16|.390295/.0979774/.00477680/.00725605|.243861/.0836104/.00767921/.0104859|.624812/.853364/1.607605/1.445123|.670352/.870750/1.602048/1.450405|
|N20|.234166/.0523928/.00287472/.00466267|.0372104/.0958755/.00759822/.00708371|.158906/1.829937/2.643117/1.519239|.164423/1.866997/2.694357/1.528859|

The ball retains N20 H²/M²/Z²/Theta² fractions.0488540/.00288626/.0000868698/.0335101, versus N16.163710/.00795966/.000135640/.0791990. A same-ball comparison still samples different grid points and different global mode shapes. N16 Omega_min=.0026953125 versus N20.022925 is a separate outer-sampling confound. There is no uniform defect decrease and no justified order estimate.

## Chosen constraint continuation and checks

The separately named componentwise same-ray constraint comparator uses88992 strictly active/nonrecursive donor references. Its centered all-domain defect/actual H/M/Z/Theta ratios are7.59596/97.51776/5.99193/8.48192. The M ratio is smaller than the N16 chosen comparator1093.72, but this extension is not the constraint closure induced by native primitive continuation. Neither value diagnoses the native boundary's stability. Strict and extended callbacks agree bitwise wherever their evaluated rows overlap.

Actual native RHS agrees with Jv to3.5898e-8 in state generator units at max-component perturbation3e-5; its C_h action agrees with cached C_hJv to≤1.71e-7 relative across both tested amplitudes. Four native constraint amplitudes agree to≤5.20e-9; propagated K_c sensitivity is≤3.98e-8. C_h(Jv−lambda*v)/C_hJv is≤8.07e-7, far below the strict mismatch. Constant/quadratic analytical-jet subsidiary oracles pass with scale-normalized error≤5.82e-16; Lx and KO annihilation is roundoff. These are empirical bridge/sensitivity checks, not exact discrete closure.

The limited reduced export took1.479s, callback build2.215s, and final diagnostic4.146s after startup. Exact compiler flags, static-library and every dependency hash are in build-provenance.json. Newly compiled production headers match27c19d20. A missing fresh inputs directory caused one preparation failure before compiler launch; its log and correction are retained, with no scientific output. All source/export/history comparisons are read-only and hash-verified.

The result reinforces a finite-grid constraint-compatibility concern for the sampled modes. It does not distinguish Hessian/product-rule/projector/diagnostic defects uniquely, prove an autonomous interior unstable system, clear the outer closure, or classify a continuum subsidiary branch.
