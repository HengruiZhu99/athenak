# Early-window V4 field diagnostic

This is a diagnostic acquisition from the already failed V4 scheme, not a new
stability trial or an admission to restart. The duration evidence is unchanged:
epsilon-two surface dissipation did not stabilize n16 Minkowski evolution.

Three independent fresh analytic-Minkowski runs stop at coordinate times 0.25,
0.30 and 0.35 (500, 600 and 700 nominal steps). All use n16, dt=0.0005,
degree seven and epsilon two, frozen source-v11 commit
`d98e2a99bbf1d5546956be3d724c82715ca704ec`, source-manifest SHA256
`0e7967f1f33559ed79b9f97d6416dacc296f0f14663d35c682177a9ac07db60f`,
and binary SHA256
`fc8d8074351a01ade6fd05dcf460555c935cf4ffe66c9bda31cc517679500fff`.
The existing epsilon-two GPU preflight admission is reused. No source, binary,
coefficient, timestep, initial datum or boundary rule is changed.

The acquisition checks require a complete clock, current finite diagnostics,
positive geometry variables, roundoff null/algebraic residuals, the original
CFL bounds, and a state deviation between 1e-10 and 1e-3. This amplitude window
is the same one used by the prior descriptive growth fit; it is a screen for
studying small errors, not proof that nonlinear terms are negligible. A failed
acquisition stops the worker. The original equilibrium stability thresholds
remain state <=1e-8 and H/M <=1e-7, and will be replayed separately on every
report, including violations. No checkpoint from this diagnostic is admitted
for scientific continuation. In particular, the native header's admitted flag
only records absence of a native exception; it is not the coordinator's
stability decision.

The predeclared analysis measures normalized state-direction correlation and
best-fit amplification between adjacent and end snapshots, the corresponding
shape residual, field and radial participation, and degree-seven B/Q response.
Use the independent tangent coordinates (26 per volume point, 19 per surface
point), with metric/A Frobenius weights, and state each area/volume weighting
explicitly. Also retain raw-coordinate results so that no norm choice is hidden.
The third time tests persistence inside the interval. Endpoint alignment alone
is not a certified eigenmode. Angular harmonic content and any native Jv replay
must retain approximation/residual checks; terminal nonlinear patterns must
not be substituted for this window. A surface filter or volume collar will not
be adopted solely from a Rayleigh estimate.

All three cases run sequentially within one 20-minute debug allocation using
CompactBinaryMerger, one node, twelve ranks and twelve GPU tiles. Each case is
bounded to 300 seconds, the worker to 1110 seconds. The plan reserves 220 MB of
new output plus a 100 MB atomic-checkpoint reserve under the campaign's 5 GB
cap. Retain hashes, small reports and histories; archive full checkpoints
locally before any verified remote purge. No failed checkpoint is restarted.
