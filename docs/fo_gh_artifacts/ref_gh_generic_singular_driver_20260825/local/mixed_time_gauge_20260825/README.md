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
- Existing dynamic-spatial, flat/nonflat source, cache, and exact-Minkowski
  oracles still pass.
- Serial and four-thread Kokkos OpenMP source/cache runs pass.
- Smooth lapse and moving-spatial references each completed one full RK4 cycle
  with gauge subtraction and wrote binary64 full-state outputs.
- The generic reference completed one full RK4 cycle with the validation-only
  centered time oracle disabled and wrote binary64 full-state output. This is
  a finite smoke result only, not stability or convergence evidence.

## Preserved limitation

At exactly `t=0`, the generic prescribed transition uses a clamped quintic
smoothstep. It is `C2`, but a centered fourth-order validation stencil crosses
the clamped endpoint and is not a fourth-order derivative oracle there. The
strict comparison fails with scaled error `2.65333e-06`; this failure is
preserved in `serial_generic_t0_centered_oracle_expected_failure.log`. The
threshold was not weakened. A gate at the interior time `t=4M` passes.

The one-cycle robust-noise summaries contain large growth factors because the
improved gauge driver deterministically moves the gauge fields away from their
reference-only initial values. They must not be interpreted as a robust
stability test, a puncture result, or convergence evidence.

## Puncture-mask rule

Primary puncture norms remain subject to the previously committed conservative
full-stencil exclusion: a sample is discarded whenever any contributing
finite-difference/evolution stencil overlaps the puncture support. For
fourth-order Ref-GH with nonzero KO dissipation this is the full `+/-3`-cell
support. This checkpoint does not alter or re-evaluate those puncture results.
