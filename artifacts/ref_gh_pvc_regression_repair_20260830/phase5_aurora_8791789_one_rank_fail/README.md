# Phase 5 one-rank discriminator

Aurora job `8791789` ran exact first-bad `ab30fa96` with the default device
image on one rank and one PVC tile. The full 96^3 grid and all 216 MeshBlocks
were retained on that rank, so the physical-boundary kernels were unchanged.

The run completed all four RK-stage RHS, update, communication, prolongation,
and physical-boundary fences, then reproduced two PDE-level Level Zero
`NotPresent` writes. PBS exit status was 134 and no positive-time history was
written. Executable SHA-256 was
`a3ec043b7c2228c1a2e92a72468a03d2e8db73c829a3db9022975c629aa20671`.

Therefore off-rank GPU-aware MPI is not required for the regression. The
explicit completion controls remain a separate device-page-lifecycle test.
