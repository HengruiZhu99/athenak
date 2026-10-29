# Cross-face Minkowski duration discriminator

This test requires a passed, independently verified source-v10 preflight.
It starts fresh analytic Minkowski data on the same nineteen n16 Cartesian
and shell blocks, uses nominal dt=.0005, targets coordinate t=1, and records
diagnostics every .05. It tests whether cross-face angular scri derivatives
suppress the growth measured in the original and projected controls.

The frozen source-v10 manifest is
`86b602e7e68fc93e6a76e16c9bd5c4aea9fcef64363670c5f95c6b0adf30fc13`,
from numerical/source commit `ec0dd5afcf9d489e06c1fdf92e802a6fb8c14679`.
The SYCL/MPI executable SHA256 is
`a557b2090b8c41c6999969a008f1731e743eea21caa3fa257af386ab15b1847d`.
It combines the previously tested final-stage algebraic projection and
cached scri jets with the centered cross-face angular reconstruction.
J remains disabled and the original gauge coefficients are retained.
Surface-to-ghost value interpolation is unchanged; the new operator changes
the derivative jets used in the surface equations. No failed checkpoint
is restarted.

Require completed evolution to t=1, current diagnostics, positive lapse
and chi, maximum state deviation <=1e-8, H/M RMS <=1e-7, both null residuals
<=1e-12, determinant/trace residuals <=1e-12, and recorded CFL <=.5.
Require the V3 schema and cross_face_centered5 marker. None of the original
scientific limits is loosened. The frozen runner accumulates time in binary
floating point: 2,000 nominal steps reach .9999999999999453, followed by one
endpoint remainder of about 5.47e-14. The expected count is therefore 2,001
accepted steps at exact final t=1, including that bookkeeping remainder.

Use CompactBinaryMerger, one debug node and twelve ranks on twelve distinct
GPU tiles. The allocation is twenty minutes, the worker budget 1,110 seconds,
and the numerical case limit 1,050 seconds (native deadline about 1,020).
Reserve 130 MB for new outputs plus 100 MB atomic-checkpoint work inside
the 5 GB home-campaign cap. Preserve any rollback and failed-stage payload
and distinguish a walltime checkpoint from completed scientific validation.

A pass establishes only this bounded Minkowski duration. Puncture stability,
the requested 100 M transition, convergence and production AMR remain
separate required work.
