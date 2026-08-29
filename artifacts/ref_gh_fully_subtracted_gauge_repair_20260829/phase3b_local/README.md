# Phase-3b residual gauge-source isolation checkpoint

This checkpoint diagnoses, rather than erases, the red result preserved in
`phase3_local`.

The earlier perturbed Einstein-source comparison supplied the residual source
with the direct residual driver time derivative but supplied the legacy source
with the independently reconstructed full driver time derivative. The latter
contains the singular binary64 cancellation under investigation. Consequently
the reported source discrepancy was not an isolated source test.

With both source paths supplied the same-stage regular time derivative,
`d_t H = d_t Href + d_t deltaH`, the complete source-unit executable passes at
the unchanged `1024*epsilon_binary64` tolerance. The all-radius source
discrepancy is `6.96333e-09`; at the conditioned radii at least `0.8M`, the
combined strict gate is `3.82012e-14`. The raw full-driver reconstruction is
retained separately and disagrees by as much as `1.05256`.

An independent 80-decimal-digit implicit-trumpet oracle, which does not read
the generated binary64 table, verifies over the expanded ten-radius matrix:

- `max |Fref-Href| = 7.595510244205564e-75`;
- `max |conformal Gamma ref| = 4.761790561071987e-80`.

This proves the exact static identities used by the matched `q=1` residual
branch. It does not yet qualify the full perturbed residual driver at every
radius, the moving-reference forcing, production dispatch, or evolution.
