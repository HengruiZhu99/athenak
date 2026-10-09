# Held compact-radius root comparison

This is a fresh SOURCE-ONLY arithmetic acceleration candidate and a separate 480-ray saved-output comparison pilot. Nothing here has been imported, syntax checked, compiled or scientifically executed during preparation. The preserved full/v2/timing bundles and completed timing attempt remain byte unchanged. The full 493568-derivative-ray gate remains HELD; it must not be launched from a ray-count runtime extrapolation. There is no inverse map, PDE propagation, native query, gauge adoption or BH inner choice.

## Exact target equation and branch proof

Write physical source radius q(r)=r/Omega(r), H(r)=q(r)+D_h(r), v=H_q=b/h, q_r=L/Omega^2>0. At fixed future inertial event (T,X) and fixed future null vector k=(k0,ksp), define

    lambda(r)=[T-H(r)]/k0,
    y(r)=X-lambda(r)ksp,
    F(r)=q(r)-|y(r)|.

For |y|>0 and nu=y/|y|,

    lambda_r=-H_r/k0=-q_r v/k0,
    y_r=q_r v ksp/k0,
    F_r=q_r[1-v nu.ksp/k0]=q_r K/k0>0,
    K=k0-H_i ksp_i.

The strict positivity uses Omega>0, h^2=b^2+Omega^2, |ksp|=k0, and b>=0. This is an analytic target identity, not a validated interval enclosure of floating quadrature. In the exact Cartesian core H=0, F_r=1 even at y=0; test the exact core root lambda=T/k0 and |X-lambda ksp|<=r0 before any source normal/division. At initial events lambda=0 is exact. The old exact outer hyperboloid root is retained only if the *outward lower endpoint* already lies at or beyond r1.

For event R_e=|X|, u=T-R_e, the cone triangle condition is |R_e-q|<=T-H<=R_e+q. Its compact source endpoints use the same stable defect equations as the preserved coarea source_bounds. If u<0, the lower equation is D_h=u; if u>=0 it is 2q+D_h=u. The upper equation is 2q+D_h=2R_e+u. At the lower endpoint F<=0 and at the upper F>=0 by the triangle inequality, independently of ray direction. A midpoint approximation may exclude a parallel ray, so the new helper retains lower *lo* and upper *hi* from the two endpoint bisections. It checks F signs and nonnegative transformed lambdas directly for each ray. At R_e=0 both endpoint equations reduce to 2q+D_h=T; their retained bracket still encloses the direction-independent source radius.

The old source_bounds/r_of_q/height code and height quadrature are not changed. New endpoint bisection never evaluates reference at r=1, retains a finite hi<1, keeps max512 iterations, and keeps the old absolute residual1e-40 and radius-width1e-50 gates. Endpoint context is shared by rays for an event in a bounded cache of8, as before.

## Safeguarded arithmetic, no tolerance relaxation

Try at most16 safeguarded Newton iterations on compact r, using the exact analytic derivative above. Outside the retained bracket, use its midpoint. Around each candidate, probe a fixed delta=tol/[8 max(1,|H_r/k0|)], clipped to the existing bracket. Accept only when direct F_left<=0<=F_right, radius width<1e-50 and the actual transformed lambda width<1e-50. The returned lambda is the midpoint of those transformed endpoint lambdas. Its original residual |T-lambda k0-H(r_of_q(|X-lambda ksp|))| must remain<=1e-40. No compact F residual replaces this test. If Newton cannot supply such a certificate, use fixed compact bisection with the original512 cap and the same three acceptance gates. Nonfinite data, bad signs, nonpositive derivative/denominator or exhausted caps fail; there is no retry or adaptive quadrature.

All derivative, pulse source, factored denominator, implicit jets and local-identity formulas stay in byte-exact copies of derivative_core.py4ba5538c..., analytic_jets.pyb2defe45... and values_context.py89c96ee.... Their mains are not called. The new module only subclasses NativeGraph.root and records arithmetic certificates. Source/source-radius inverse and graph residual still use the original context. This removes nested physical-radius inversion from the repeated transition-root loop; it does not claim a measured speedup until the separate pilot runs.

## Fixed pilot and saved cross-binding

Use exactly the completed timing recipe's three events, two constant boosts, 60/80 digits, levels2x4/height32 and4x8/height64, amplitudes.2/.1,width.35,S1/a.5,geometry.05-.95. The original timing PASS is bound by its child/outer receipts and all480 saved ray records, groups and checks. Its recorded193.8207s child time is a measured old-pilot fact, not an estimate of the new or full gate.

Recompute exactly480 local ray jets,24groups and the original4744 checks. Each key (digits,level,event,boost,polar_index,azimuth_index) must match exactly one original row. Preserve all60 scalar/value/gradient/ten-Hessian entries per ray and compare them individually at the unchanged precision threshold1e-30. Compare fixed k at the unchanged local threshold1e-35. Compare root metric at1e-40; the other six local metrics and minimum_D at1e-35, using max(1,|old|,|new|) scaling. Retain the original positivity guard for minimum_D. Compare the new two precision grids as before. The old rays did not save lambda, so this pilot cannot assert lambda agreement with an unavailable original value.

Counts are4744 original checks plus per-ray60jet+1k+8metric+1compact-certificate checks:38344 total. There are no new angular-accuracy claims or thresholds. The added compact certificate records method, retained endpoints, signs, both widths, original residual and endpoint/Newton/fallback iteration counts where applicable. Exact initial/core/outer branches record their original residual and analytic branch. Append each completed ray immediately to completed-rays.jsonl, so mid-group failures retain completed rows; original snapshots remain untouched. Save full rays/groups/checks at completion and keep every partial/failure.

The exact root recipe remains local-path/hash bound. Authorization must pin all seven consumed sources/recipe/PLAN, the fresh output, runtime and dependencies. The one-shot standard-library outer launcher is separate. It captures source-before copies, true process status, full stdout/stderr, before/after dependency and saved-output hashes, and refuses reused paths. No execution is admitted by this plan.

## Limits

This pilot tests local analytic derivative identities and arithmetic equivalence on the old fixed ray sample. It does not certify angular quadrature convergence, all future events/ray directions, exact interval root enclosures, inverse-map/Jacobian regularity, target-time coverage, continuum gauge stability, native pulse acceptance or wormhole-to-trumpet behavior. Group timing excludes graph/height-prefix construction; local seconds include the root/source/jet work and first-event context setup. The overall child and outer receipts also report elapsed time. Keep failures and original gate classifications without tuning.
