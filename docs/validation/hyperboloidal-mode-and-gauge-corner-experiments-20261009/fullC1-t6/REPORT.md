# Full C1 spatial-norm longer-window screen

The previously frozen full C1 plus covector-Lambda candidate does not remove the large late growth seen in the original C0 fixed-grid projected-continuous operator. A fresh exploratory N16 t6 action of its original matrix gives gauge-seed H/M/Z ratios to C0 of1.006/1.186/.961 and shell-seed ratios1.179/1.463/1.138 at t6. This answers the longer-window question left open by its modestly worse t2 transient. It is not a nonlinear native run, a long independent canonical check, a physical energy bound, a certified eigenvalue test, or a continuum/scri stability result.

The original fullC1 sources, exes, matrices, seeds, local scientific gates, short canonical and native preflight evidence remain unchanged. This extension uses only the original spatial-norm C0 gauge with full mechanical C1 plus the separately derived covector Lambda repair. It adds no lapse modification, inner C1 blend, prescribed/live damping profile, stencil change, ghost change, or runtime projection change.

## Pinned operator and calculation

N16/span2.2 has1640 active points and32800 free20 coordinates. S=1,a=.5, geometry(.05,.95), kappa input10, kappa2=0, symmetric quadratic ray ghosts and native KO=.1 are unchanged. Its minimum Omega is .0026953124999994555, h=.1375, and nominal pole dt8.085937499998367e-5. The action here is `exp(t P_ref J22 Lift_ref)` at fixed grid, with continuous algebraic restriction. The exact native final-only RK3 tangent is a separate map already checked by the original gate; no long native step integration is performed here.

The reused C1 projected matrix SHA is `1093d7c07a71019cd69e578d52ca0c47635dc190b32ccb371683fd162132241e`. The prior completed C1 manifest is `5a44232529dd90e71f493895a6742817d73ee016066ab0cdead159dd19cdee4b`, whose source-at-launch is aef47b0a and public runtime implementation27c19d20696ea6dd4704032c51dfd026218f64f2 plus explicit frozen overlays. The finite-Omega C1 correction is used after the existing C0 reference subtraction without subtracting a C1 reference RHS; its original raw-reference cancellation and double-pole/transient caveats remain applicable.

Both original gauge and radial-shell/random seeds are separately normalized to unit free20 Euclidean norm, exactly as in the original t2 calculation and C0 t6 comparison. The matched physical gauge seed can be recovered by the original seed scale; this extension does not renormalize across candidates. Saved times are0..6 at cadence.025 (241 times, two seeds). The original outward-reference crossing time is .7457643839234269, so t6 is about8.045 crossings of that stationary reference.

The fresh double-reorthogonalized Arnoldi calculation uses dimensions50/80, local comparison tolerance1e-10, maximum interval.1, actual Krylov-curve residuals, and a1e12 Euclidean-amplification/nonfinite guard. The guard did not trigger; all saved vectors and scalar JSON receipts are finite. NumPy emitted an invalid-matmul RuntimeWarning during this run; its exact text is retained in run.log, with no assigned cause. The finite-state, direct-curve, epsilon and independent-prefix checks below passed. Local truncation differences and small curve defects do not provide a rigorous nonnormal forward-error bound.

## Endpoint and peak measurements

Native H/M/Z are unweighted active-cell RMS, with M/Z covectors contracted using the stationary reference Penrose spatial inverse, matching the original actual native diagnostic callback. The component-unit norm is the original reference-volume integral of configuration values/gradients and momentum values; it uses stored Cartesian components and is **not a physical energy or proved symmetrizer norm**.

| Seed, t6 | C0 H | C1 H | C0 M | C1 M | C0 Z | C1 Z | C0 component amplification | C1 component amplification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gauge pulse | 5801.45426 | 5836.56200 | 7449.88930 | 8833.88422 | 2504.38813 | 2405.71995 | 97061.85023 | 94987.38717 |
| Shell/random | 5.16092401 | 6.08660116 | 5.19134922 | 7.59426786 | 1.38165718 | 1.57220178 | 23.85252384 | 26.52883496 |

The gauge H/M sampled maxima are at t6; its Z sampled maximum is2511.61524 at t5.95, followed by2405.71995 at t6. Both seed component-unit norms have their sampled maxima at t6. The trajectories oscillate: endpoint growth does not imply every sampled constraint is monotone. The complete histories and sampled peak values/times, including the shell constraints, are retained in summary.json and the native analyses.

At t6 the squared outer r>=.9 fractions are gauge H/M/Z .02923/.89103/.87092 and shell .04294/.88666/.70614. H peaks at r=.83354945 for both seeds. Gauge M/Z peaks at the closest shell r=.99865143; shell M peaks there and Z at r=.91981231. This is localization of the existing finite-grid tangent data, not a causal boundary attribution or a mode identity with C0.

## Numerical and provenance checks

The fresh trajectory agrees with the frozen original C1 t2 action over all81 shared times and both seeds to a maximum relative state error3.878419542424415e-14. No original t2 trajectory was rerun or rewritten. The new long-state SHA is `e9a8fbf271cb2ec67e23d7071d46f8afe2b544af2630ffda30d6f7bcca3ed62b`.

Propagation took188.7498s, with17840 Arnoldi matrix products and703 extra direct residual products. The maximum local50/80 relative action difference is9.835163258313364e-11; the maximum actual curve defect divided by state norm is2.825220878923307e-10 per unit time. Actual native constraint directional derivatives at perturbation sizes1e-5 and3e-5 agree to at most1.299e-9 relative over the selected initial/mid/final checks. Native H/M/Z and field diagnostics took7.7550s and3.2422s. Original compiled oracle bytes, dependencies, archives, overlays and completed archive are checked again by freeze_long.py.

No accepted generator eigenvalue, global-rightmost growth rate, long canonical action, or long nonlinear native outcome is inferred from this extension. The evidence supports continued rejection of this fullC1 candidate as a demonstrated remedy for the original coarse-grid long-window growth; higher-resolution and actual finite-step behavior remain distinct questions.
