# Analytic mixed-time gauge-reference checkpoint

This directory contains compact local CPU evidence for the equation-preserving
extension that permits gauge-reference subtraction with a smooth
time-dependent reference. Large binary field outputs remain under `/tmp` and
are not committed; their sizes and SHA-256 hashes are recorded in
`field_output_hashes.txt`.

The current checkpoint text is also incorporated in
`docs/ref_gh_generic_singular_driver_20260825.md`. The sibling PDF remains the
prior checkpoint because the required PDF-operation marker was unavailable in
this environment; it was not silently regenerated from stale TeX.

## Implemented slice

- `ReferenceJet` now carries only the twelve required mixed derivatives
  `d_t d_i d_q`, rather than a full third-order jet.
- Product, reciprocal, logarithm, exponential, and square-root operations
  propagate that slice analytically.
- Provider storage grows from 64 to 100 Reals. Symmetric metric storage reduces
  the update workspace from 416 to 410 Reals. The 313-Real hot evolution cache
  is unchanged.
- Staged kernels form `d_t d_i Href` and then the analytic derivative of
  `theta_ref=-beta^i d_i Href-(Omega_t-beta^i Omega_i)Href`.
- The evolved `delta theta` RHS subtracts this analytic `d_t theta_ref`.
- The former blanket fail-close for time-dependent gauge subtraction is
  removed only after the focused analytic gates below.

No GH equation, gauge-driver target, damping parameter, finite-difference
operator, puncture stencil mask, or reference profile was changed.

## Passed local gates

- Closed-form mixed-time jet algebra: maximum error `2.22045e-16`.
- Smooth lapse reference, all tested RK stages: analytic `d_t theta_ref`
  versus an independent fourth-order time-difference oracle, maximum observed
  scaled error `4.94e-14`.
- Smooth moving-spatial-frame reference, all tested RK stages: maximum observed
  scaled error `2.89e-15`.
- Generic singular reference at `t=4M`: scaled error `6.70727e-13`.
- Generic singular reference at the clamped `t=0` endpoint: an endpoint-aware
  fourth-order forward oracle passes with scaled error `1.77e-17`.
- Existing dynamic-spatial, flat/nonflat source, cache, and exact-Minkowski
  oracles still pass.
- Serial and four-thread Kokkos OpenMP source/cache runs pass.
- Smooth lapse and moving-spatial references each completed one full RK4 cycle
  with gauge subtraction and wrote binary64 full-state outputs.
- The generic reference completed one full RK4 cycle with the endpoint-aware
  oracle enabled in Kokkos Serial and four-thread OpenMP, and wrote binary64
  full-state output. The earlier validation-disabled cycle remains preserved
  as provenance. These are finite smoke results only, not stability evidence.

## Preserved limitation

At exactly `t=0`, the generic prescribed transition uses a clamped quintic
smoothstep. It is `C2`, so a centered fourth-order validation stencil crosses
the clamp and is not a valid fourth-order derivative oracle there. The original
strict centered comparison fails with scaled error `2.65333e-06`; that failure
is preserved in `serial_generic_t0_centered_oracle_expected_failure.log` and
its threshold was not weakened. The corrected validation selects a fourth-
order one-sided stencil only when a centered stencil would cross a prescribed
clamp. It passes at `t=0` and returns to the centered stencil at interior RK
stages.

A fixed-grid three-level temporal ladder through `t=0.008M` independently
checks the evolved result. Because AthenaK uses adaptive final steps, the
self-difference ratio is normalized by the actual variable-step RK4 leading
error weight `sum(dt^5)`. The norms discard all 64 native cells whose active
fourth-order finite-difference support box overlaps the puncture, retaining
152 of 216 cells. (KO is disabled in this discriminator, so the active radius
is two cells.) Effective orders on that puncture-clear set are
`4.003`/`4.006` for Einstein L2/Linf and `3.987`/`3.983` for gauge L2/Linf.
Thus the clamped endpoint does not measurably degrade this bounded RK4 test.
This remains a short temporal discriminator, not puncture stability.

The one-cycle robust-noise summaries contain large growth factors because the
improved gauge driver deterministically moves the gauge fields away from their
reference-only initial values. They must not be interpreted as a robust
stability test, a puncture result, or convergence evidence.

## Puncture-mask rule

Primary puncture norms remain subject to the previously committed conservative
full-stencil exclusion: a sample is discarded whenever any contributing
finite-difference/evolution stencil overlaps the puncture support. For
fourth-order Ref-GH this is the full `+/-2`-cell support with KO disabled and
the full `+/-3`-cell support with nonzero KO dissipation. The temporal result
above was recomputed with this rule; no unmasked puncture cells enter its
reported norms.
