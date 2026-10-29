# V4 angular-dissipation Minkowski duration comparison

This is a bounded stability test of the optional degree-seven, epsilon-two
scri operator after its independently verified GPU preflight. It is not an
admission of stable puncture evolution, convergence or production AMR.

Run fresh analytic Minkowski data at n16, dt=.0005, through coordinate t=1,
with diagnostics every .05. The 19-block Cartesian-core/spherical-shell grid,
bulk coefficients, V3 cross-face scri jets, final tensor projection, original
volume KO and geometric regularity terms remain unchanged. V4 adds only its
specified angular rates before dependent-D reconstruction. No symmetry
projection, metric floor or positivity repair is added to ghost interpolation.

Keep the previous V3 duration error limits: state deviation <=1e-8, H/M RMS
<=1e-7, surface/ghost null and accepted algebraic residuals <=1e-12, positive
chi/lapse, bulk CFL <=.5 and current diagnostics. The separate isolated-Q
bound is <=2.7; passing it does not prove coupled stability. Require completed
t=1 with 2001 accepted steps, following the existing floating-point clock
bookkeeping including its final endpoint remainder. A failed or walltime-
bounded result is not a completed stability pass. Native checkpoint admission
alone does not override failed scientific limits.

The source is frozen commit `d98e2a99bbf1d5546956be3d724c82715ca704ec`, full
manifest SHA256
`0e7967f1f33559ed79b9f97d6416dacc296f0f14663d35c682177a9ac07db60f`.
The V4 SYCL binary SHA256 is
`fc8d8074351a01ade6fd05dcf460555c935cf4ffe66c9bda31cc517679500fff`.
The epsilon-two [preflight admission](../preflight-dissipative-v1/admission.json)
has SHA256 `32cc0fed05319e204f41edea3b62f614138f07a801079e6348536a07650713a1`.
The reviewed plan SHA256 is
`daf10ec4f22cfd961c27582895fb9d1e5cfb205f7d3f71458c6882185f8207f2`.

Use one CompactBinaryMerger debug node, twelve ranks and twelve GPU tiles,
with a twenty-minute allocation, 1110-second worker and 1050-second case cap.
Reserve 130 MB of new outputs and a further 100 MB atomic-checkpoint margin
under the 5 GB campaign cap. Preserve reports, histories, source/input hashes,
the accepted rollback checkpoint and any failed stage before analysis.

Compare the recorded trajectory with the original, projection-only and V3
controls. Any growth fit must state its actual samples and interval. The
previous amplitude window, state deviation 1e-10 through 1e-3, remains the
selection rule; insufficient samples give an unresolved rate, not zero growth.
Stop plotted diagnostics at their recorded times instead of extending stale
values to a later failure endpoint. Report null closure separately from field
and constraint stability.
