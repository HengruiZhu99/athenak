# Read-only external review prompt

Repository: <https://github.com/HengruiZhu99/athenak>

Branch: `codex/z4c-vc-hardening-2d3d-20260822`

Compiled production source authority:

- commit: `99a4eb5ba7713f7de73239cf75a27c1fb9ac6cbb`
- tree: `e8c1083cc9ea67aa4a3a2c3adbffb9c31fe32c83`
- Kokkos: `6739bc623081648af9e752b616d9671527922cbf`

Use the pushed branch tip named in the accompanying handoff for the final
reports. Do not substitute a later branch state.

Start with:

- `docs/investigations/z4c_vc_hardening_20260822/REPORT.md`
- `docs/investigations/z4c_vc_hardening_20260822/ROOT_CAUSE.md`
- `docs/investigations/z4c_vc_hardening_20260822/TEST_MATRIX.md`
- `docs/investigations/z4c_vc_hardening_20260822/BACKEND_STATUS.md`
- `docs/investigations/z4c_vc_hardening_20260822/CC_REGRESSION.md`
- `docs/investigations/z4c_vc_hardening_20260822/EVIDENCE_MANIFEST.json`

Please perform a skeptical, read-only source and evidence review. Do not run
jobs or modify the repository.

Established facts to audit:

1. The original Aurora/SYCL page fault came from
   `MeshBoundaryValuesVC::ProlongateVC` capturing host-resident
   `recvbuf[n].iprol[0]` inside a device lambda. Commit `aa668dd0` moves those
   bounds into a synchronized device DualView. The repaired A0--A19 and
   A150--A159 canaries pass repeatedly.
2. Host and exact-final PVC O2/O4/O6 constant-state refine/derefine lifecycle
   cases pass in 2D Cartesian, 2D axis-touching Cartoon, and 3D Cartesian.
3. A nonconstant O4 AMR wave discriminator is not convergent: 2D observed
   order `-0.9490`, 3D order `-0.4459`. The first substantial mismatch appears
   after the first evolved fine stage at an interface. A speculative authority
   synchronization was reverted after N64 divergence.
4. Host restart/replay/output paths pass. Exact-final SYCL 3D continuation from
   a post-refinement checkpoint produces a different final payload despite
   matching time, timestep, and hierarchy. The two equal-length payload hashes
   are
   `e79600535d266b13b06fefec6aec2b80d51354bf0b5875e8533c4dd29a073217`
   and
   `131e6929ef1397916b191d6963c02561f0f107f615dabd5493cb2614cdc26e69`;
   histories differ only near roundoff.
5. Historical cell-centered payloads remain exact. Current-source CUDA runtime
   evidence is unavailable. The common-tree Brill evidence remains
   `O4_NONCONVERGENT` and must not be upgraded.
6. Aurora job `8775539` is only a failed Python harness retry: NumPy was
   available but `h5py` was not, so it produced no new numerical restart or MPI
   result. Do not count it as a physics/code failure. The subsequent narrowly
   scoped job `8775545` passes exact-final PVC two-rank 2D Cartoon O4 and 3D
   Cartesian O4 lifecycle cases.

Please answer:

1. Is the host-pointer-capture root cause and DualView repair complete and
   correctly scoped for all VC prolongation consumers?
2. Is there any remaining device closure that captures host metadata, an STL
   owner, a stale topology object, or an execution-space dependency that the
   current audit missed?
3. What is the smallest source-grounded explanation for the negative
   nonconstant AMR convergence after constant-state and analytic fixed-grid
   gates pass? Examine shared-node authority, coarse/fine time-state freshness,
   vertex ownership, prolongation/restriction composition, and stage ordering.
4. What is the smallest decisive diagnostic for the SYCL-only 3D restart
   payload mismatch? In particular, which first post-restart phase/state slice
   should be compared, and could omitted ghost/coarse/topology state explain
   why host is bit-exact while PVC is not?
5. Does any cell-centered path change floating-point operation order or restart
   schema despite the exact regression evidence?
6. Recommend at most one narrow diagnostic or source correction for each
   unresolved failure. State what observation would falsify it.

Do not recommend floors, clipping, weakened chi/admissibility gates, broad
parameter sweeps, gauge/KO/CFL tuning, or unsupported convergence/physics
claims. Keep observation, inference, and hypothesis explicitly separate.
