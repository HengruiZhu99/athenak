# Phase 3 Hybrid A

Hybrid A overlays the fully-subtracted `ab30fa96` RHS and its required helper
headers on W while retaining W's task and stationary problem-generator blobs.
The exact overlay manifest SHA-256 is
`3511e55316a7e0c2467b7d0dd7cf48d9d266f9885982d40a9800ce2db13fcf9f`.

Aurora job `8791763` reproduced fourteen Level Zero PDE-level `NotPresent`
writes on node `x4216c4s7b0n0`; PBS exit status was 143 and no positive-time
history was written. The executable SHA-256 was
`4068d6e7f1574f938a11c96fde9b2cf3bd78b2c1100aa66fa1762bf5b48e340b`.

Hybrid B is the already-executed `b290c2bd` midpoint: its production RHS blob
is byte-identical to W while it contains the repaired exact-state/task/boundary
path. It passed. The matrix therefore associates the regression with the
fully-subtracted RHS/helper device image, not boundary semantics.
