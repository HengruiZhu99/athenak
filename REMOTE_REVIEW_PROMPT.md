# Read-only review request: Z4c AMR coarse-cache ownership repair

- Repository: https://github.com/HengruiZhu99/athenak
- Branch: `codex/brill-amr-coarse-cache-coherence-20260817`
- Repair commit: `ab651f0ebd113f8718fefbf6d802976e6b3e8738`
- Report: `docs/investigations/brill_amr_coarse_cache_fix_20260817/REPORT.md`
- Evidence directory: `docs/investigations/brill_amr_coarse_cache_fix_20260817/`

Please perform a skeptical, read-only source and evidence review. Do not run jobs or edit the repository.

## Established observations

The captured cycle-1722 Cartoon N256 event had correct same-level fine ghosts and correct same-level `coarse_u0` immediately after receive. The old `FillCoarseInBndryCC` then overwrote owner-derived received values using receiver-local high-order restriction.

The exact writer/consumer audit found 7,056 unique coarse cells consumed by O6 prolongation: 4,736 supplied by a coarser receive, 1,624 by local active restriction, 672 by same-level receive, and 24 by physical-boundary construction. None were missing or multiply authoritative before the refresh; 344 consumed cells were redundantly overwritten.

The old production P5 replay matched 201,600 captured fine values to `6.66e-16`. Preserving received cache bytes changed 6,191 consumed coarse variable values and 44,261 downstream fine values.

The repair skips same-level local refresh only for Z4c. Generic finite-volume fields retain the old path. NGHOST 2/3/4, all 2D face and child orientations, axis/off-axis, same-rank/MPI2, all 25 variables, and a 3D case are covered.

Exact patched zero-PDE job 57168348 passed. Its post-BC and nominal-refresh `u0`/`coarse_u0` files are byte-identical; sender/receiver and cache mismatch flags are false; chi remains finite and positive.

Nevertheless, the authoritative C++ cycle-1722 constraint jump is unchanged:

- old C² factor `6.23477935`, patched `6.23478468`;
- old H² factor `22.9319109`, patched `22.9320471`;
- old M² factor `80.6125157`, patched `80.6116325`;
- old Z² factor `1.00054175`, patched `1.00054175`.

This establishes a real cache-coherence defect but shows that it is secondary to the dominant event jump.

The single matched short N256 continuation, job 57168637, reused the exact
authenticated source/executable/restart and unchanged O6/RK4/CFL/KO, gauge,
damping, and `dchi_max=0.01` settings.  It timed out rather than reaching
`t=12.5 M`.  At the first row beyond the old terminal time it had essentially
the same C², level, and timestep as the predecessor.  It then continued the
same refine/derefine runaway to `t=11.9558403611 M`, cycle 4389, level 19,
665 MeshBlocks, `dt=3.5762786865e-8 M`, and `C²=2.3608924968e11`; the final
log begins a level-20 creation transaction.  No stabilization or production
qualification is supported.

Key v6 hashes:

- complete raw-run manifest:
  `d2dce5a4fccdfb1a97507256de5dcb3765aa60d771d988d53a683854ace012e4`;
- allocation manifest:
  `200b5d6372cf40f081aa74f3593be5e9e20bd6b92f89f8e95ba6f9c306252c70`;
- run log:
  `3044d88330d1130aad2c97f2fce66976749509a77d698d60325be6744f2cc958`;
- history:
  `8836677539f12608a6a385d3d0ff571f68e5a28f3511210809eee195b550beba`.

## Questions

1. Is the owner-authoritative Z4c cache policy correct for every relevant 2D/3D topology, or is there a legitimate missing-cell case that the inventory/regression overlooked?
2. Is the early Z4c return in `FillCoarseInBndryCC` the smallest safe implementation, while preserving generic finite-volume semantics?
3. Given the byte-level repair and unchanged production constraint jump, what is the smallest decisive next diagnostic inside the regrid transaction?
4. Which writer/phase should be instrumented next to distinguish newly created active-state transfer error, derivative reconstruction across the new hierarchy, and parent under-resolution?
5. Are any existing tests accidentally polynomial-exact or insufficiently sensitive to cache ownership, block-edge orientation, axis parity, or MPI decomposition?
6. Please identify any concrete indexing, ownership, current-stage, or boundary-condition bug visible in the relevant source.

Please pay particular attention to the smallest decisive next boundary after
the cache repair: newly created active-cell population versus the later
derivative/ghost reconstruction on the accepted hierarchy.  Assess whether a
production-C++, same-leaf constraint census immediately after those two phases,
with parent O6 and shadow O4 values, is sufficient or whether an even smaller
source-level diagnostic exists.

Recommend one bounded next diagnostic or source correction, with an explicit decision rule. Do not recommend chi floors/clipping, weaker finite/positivity gates, P8 promotion, gauge/damping tuning, broad parameter sweeps, or unsupported convergence/Figure-3 claims. Keep observations, deductions, and hypotheses separate.
