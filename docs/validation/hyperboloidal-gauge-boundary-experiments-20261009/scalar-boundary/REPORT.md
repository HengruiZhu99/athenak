# Actual spherical scalar closure isolate

All source and outputs are ignored scratch files. Native runtime/source files are unchanged.

## Equation and norm interpretation

We use q_t+x·∇q=0 on the unit Cartesian ball, so all boundary normal speeds are outward. The exact solution is q(x,t)=q0(e^(−t)x). Pointwise amplitude never exceeds the initial continuous supremum. Unweighted E=∫q²d³x obeys E′=3E−∮q²dS; e^(−3t)E is nonincreasing. A regular positive time-independent spatial L2 weight cannot contract every smooth core pulse while admitting constants: div(w x)>0 near the origin when w(0)>0. A singular radial weight r^(−p), p≥3, loses constant integrability. Therefore raw L2 expansion and positive Euclidean logarithmic norm alone are not called instabilities.

## Exact discrete operator and checks

The C++ exporter evaluates actual production Dx<3>/Lx<3>/InteriorKOSixth on raw stencil basis values, expands the actual symmetric quadratic ghost plan, and independently checks the resulting sparse matrix against actual FillSphericalGhosts plus native stencil evaluation on a deterministic field. Constants, strict inside/nonrecursive donors and supported grid admission are checked. Maximum constant residual is ≤1.83e−14; maximum independent matvec discrepancy is ≤7.11e−14. Native interior KO=.1 is unchanged across high-order comparisons.

Native span2.1 N12/N16 fails ball/halo admission; N12/span2.2 also fails degree2 normal-ray rectangle admission. These rejections are preserved. Admitted common-span controls use N16/20/24, span2.2; exact production geometry uses N24/span2.1. There is no silent downgrade.

Full dense scalar spectra are computed at N16 (1640 unknowns). N20/N24 use converged sparse LR8 ARPACK with residuals stored, not full20 matrix-free eigensolves. Matrices, leading eigenvectors, exact point coordinates and pulse histories are retained.

|Grid|ray centered leading Re λ|ray upwind leading Re λ|centered mode squared weight r>.8|
|---|---:|---:|---:|
|N16, span2.2|5.34885|1.27e-15|94.102%|
|N20, span2.2|7.3599215|5.55e-16|92.752%|
|N24, span2.2|7.6366232|-1.87e-15|95.861%|
|N24, span2.1|5.1231917|1.03e-15|96.627%|

The exact production N24 centered leading pair is5.12319±5.33816i, residual4.99e−11. It is boundary localized (96.627% squared support outside.8), while near-core support is5.51e−10. Native upwind has the preserved constant mode at roundoff and subsequent modes near−1. N16 centered has192 positive real eigenvalues with KO.1; KO0 has307, so interior KO does not cure this closure.

## Simpler boundary controls

Nearest extension uses equal positive weights on the closest strict-interior donors from each actual plan, including all exact distance ties; constants and symmetry are preserved. Only ghost extension changes. It removes the centered positive modes on all tested grids, including production N24 geometry, but it is not a high-order consistency/energy guarantee and exhibits finite pulse overshoots (up to1.20 at N20, versus exact continuous sup1).

The inward first-order fallback replaces an advection axis only when its actual centered/upwind stencil samples outside the active sphere. All other native rows and native interior KO remain. Every inward neighbor is active for this radial velocity/cell-centered cube. It also removes centered positive modes on all grids. The support-change audit proves zero change in rows whose entire centered advection stencil is inside. Boundary accuracy is lowered.

An all-domain inward first-order control with KO0 is Metzler and has row sum zero. Thus exp(tA) is row-stochastic and contracts the max norm while preserving constants. This is a defensible scalar baseline, not a proposed production downgrade.

Smooth off-axis Gaussian pulses use SSPRK3 with dt=.06h to t12. Upwind ray and both alternatives remain bounded; the centered ray pulse reaches amplitude >10^8 and stops. Recorded centered amplitudes exceed4.8e7, and normalized energy exceeds initial by ≥3.2e10, contradicting continuum max-amplitude and energy behavior. This is genuine discrete growth, not geometric dilation. At production N24, ray-upwind final Linf error8.98e−7; fallback-centered final5.29e−6, nearest-centered4.57e−5. Early underresolved pulse errors are recorded separately from late behavior.

## Scope and next action

This isolate proves that native quadratic ray extrapolation is unsafe with a centered outward first derivative; it does not show that native Lx scalar advection causes the full20 instability. It contains no zero incoming scri characteristic, coupled centered second/mixed derivatives, gauge sources, nonlinear metric algebra or variable-coefficient layer physics. Full-system transfer requires a separate characteristic/constraint analysis.

A practical next scratch test is a coupled characteristic outflow boundary row with inward/SBP-compatible derivatives, or a boundary-fitted outer shell with a discrete energy estimate. Applying nearest scalar extension blindly to metric/curvature fields is not justified. Retain native interior consistency and verify actual full20 tangent/finite pulse convergence before any production change.

## Reproduction and provenance

export_operator.cpp, build-command.json, compiler identity, static-library hashes, all project-local dependencies/source bytes and immutable executable are archived in summary.json/source-snapshot. Executed base driver is driver-source-at-run.py; exact production closure controls use run_exact_closure.py. The base driver used Path.with_suffix for stdout filenames, which collides for KO.0/.1 log names; scalar matrices and receipts were named correctly and remain unchanged. The precise JSON command results in receipts are authoritative. A logging-only correction in run_isolate.py may be used for reruns.

```sh
PYTHONPATH=build-layer-research/boundary/python-deps OPENBLAS_NUM_THREADS=1 \
  python3 build-layer-research/boundary/scalar-isolate/run_isolate.py
PYTHONPATH=build-layer-research/boundary/python-deps OPENBLAS_NUM_THREADS=1 \
  python3 build-layer-research/boundary/scalar-isolate/run_exact_closure.py
```
