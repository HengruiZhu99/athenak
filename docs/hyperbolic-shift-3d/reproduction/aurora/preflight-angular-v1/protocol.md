# Cross-patch V3 GPU preflight

The original spatial/timestep controls and the projection-only V2 duration
test all failed. The next candidate combines final-stage algebraic repair
with the previously manufactured-solution-tested cross-face angular scri
reconstruction. This preflight checks its GPU implementation before duration
testing; it is not a stability or convergence result.

Frozen source-v10 is Git commit
`ec0dd5afcf9d489e06c1fdf92e802a6fb8c14679`, with Kokkos pin
`6739bc623081648af9e752b616d9671527922cbf`. The complete 2,602-file source
manifest SHA256 is
`86b602e7e68fc93e6a76e16c9bd5c4aea9fcef64363670c5f95c6b0adf30fc13`.
A fresh SYCL/MPI/PVC build using `make -j64` completed successfully; the V3
binary SHA256 is
`a557b2090b8c41c6999969a008f1731e743eea21caa3fa257af386ab15b1847d`.
No old build directory was reused. Source-v9 was an unused intermediate
archive and was never built or admitted.

Twelve cases run on one Aurora debug node under CompactBinaryMerger with
twelve ranks/tiles: bulk, halo, original-scri, algebraic and cross-face
CPU/GPU component comparisons; four-step Minkowski; two-plus-two restarted
pulse versus four-step direct pulse; a full-array restart comparison;
four-step isotropic puncture using lmax=12/ntheta=24 FastFlow; and a standalone
communication benchmark. The evolution cases use n16, dt=.0005 and the V3
checkpoint schema. They require current diagnostics, positive geometry,
roundoff-scale null and algebraic residuals, the correct scheme marker,
and exact accepted clocks/step counts. The puncture search must be accepted
at expansion RMS <=.001. Restart payloads must match bitwise.

The communication test covers n16/n24/n32 exact ownership layouts, device,
host-only and explicit-staging routes, both added-barrier and no-added-barrier
modes, 24 repetitions after three warmups, and six mixed 4-volume/14-surface
sequences. It requires SYCL, twelve ranks, double precision, distinct host
and device spaces, and full receive-array validation. Layout-only output
does not pass. Per-rank timings remain raw evidence; no network-bandwidth
or coupled-evolution speedup is inferred from message size alone.

The reviewed plan SHA256 is
`e7a6e774a96b602471be0799b8b0fdde49f605f1e72e8ae3a9b27bc8217cddae`.
Its case ceilings total 3,350 seconds inside a 3,510-second worker and one-hour
allocation. Planned new output is 750 MB plus a 100 MB atomic-checkpoint
reserve under the fixed 5 GB home-campaign cap. Failed cases stop the plan;
all outcomes and missing downstream cases must be retained explicitly.

Only a passed, independently verified preflight can admit a fresh V3 duration
test. It cannot admit the requested 100 M transition, AMR, nonspherical
evolution or convergence by itself.
