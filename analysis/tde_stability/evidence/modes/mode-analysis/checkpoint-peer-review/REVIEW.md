# Checkpoint validity helper: independent bounded review

Reviewed `analysis/tde_stability/check_checkpoint.py` against the current writer in `src/outputs/restart.cpp`, POD declarations in `src/mesh/mesh.hpp`, mesh ordering and load balancing, analytic trumpet construction in `src/pgen/z4c_tov_ks.cpp`, and the reused `tst/regression/z4c_background_restart.py` reader.

Conclusion: **no remaining blocker for its explicitly restricted scope**, after the author added the requested `chi_psi_power == -4` guard. The reader is appropriate for these double-precision, little-endian, three-dimensional MHD+25-field residual-Z4c, direct M=R0=1 Schwarzschild trumpet checkpoints on a uniform root mesh. It supports serial output with multiple blocks, or MPI with exactly one block per rank. It is not an AMR/general production checkpoint validator.

Reviewed helper SHA256: `32840289a16c24a1a17cf1c1c8de125326b0baf43b9cb74b6ea6f579bda37709`.

Reused parser SHA256: `faa25188b55c0ed44deb318c4e2eb58955b94e537c4e60997cc4819004a91b70`.

Repository HEAD during review: `1b93538a6e0769477fe72dc169ab7cf9f0161b26`; the helper was an uncommitted addition. I changed no source/helper files. This report is the only review artifact added here.

## Header and payload offsets

Let E be the first byte after `<par_end>\n`, and N be the global block count. The writer layout matches the helper:

- Global block count/root logical level: 8 bytes.
- RegionSize: 9 double values, 72 bytes.
- Root RegionIndcs and block RegionIndcs: each19int values,76bytes.
- Time, timestep and cycle: 8+8+4=20bytes.
- Global logical locations: N times16bytes, followed by N float costs (4bytes each).
- Two Z4c output times:16bytes.
- Per-block payload stride:8bytes.

Thus the logical-location start is `E + 8 + 72 + 2*76 + 20`, and the payload start is that value plus `20*N + 16 + 8`, exactly as implemented. The parser rejects extra tracker, radiation, hydro and turbulence layouts, and verifies the per-block stride matches MHD cell values, three staggered magnetic fields, then25Z4c fields. It also verifies complete payload blocks and complete matching rank cohorts.

All payload values, not only Z4c fields, are tested for finiteness. The Z4c parser itself fails closed earlier if those state values are nonfinite. In that situation the helper raises rather than emitting a completed negative-result JSON; this does not create a false pass.

## Block mapping and coordinates

`BuildTree` orders `lloc_eachmb` by global block ID in Z order. `LoadBalance` assigns contiguous global-ID intervals to ranks. `MeshBlock` uses `igids + local_index`, and the writer emits each rank's local blocks in that same order. Consequently:

- Serial concatenation is global block-ID order.
- With exactly one block per rank, rank index equals global block ID, so concatenating rank files is also global block-ID order.

The helper enforces exactly one block per rank for MPI. It also requires every saved logical level to equal the saved root level, in addition to `refinement=none`. Under those conditions its uniform coordinate formula, including all ghost centers, is the same mathematical grid as the mesh constructor. The stored ghost-depth samples correctly identify fourth-corner ghosts in the failing case. Ghost counts refer to stored block cells, so shared ghost coordinates can occur in multiple blocks; they are not unique physical-cell counts.

## Metric reconstruction and one corrected guard

The source's direct trumpet has conformal metric I, lapse `r/(r+1)`, and chi `((r+1)/r)^(chi_psi_power/2)`. The helper uses chi=`(r/(r+1))^2`, which is correct only for exponent-4. I identified the missing exponent guard; the author added it before final review. All three independently checked datasets already used-4, so this does not alter their conclusions.

For a real symmetric conformal metric, positivity of gxx, the leading2x2minor, and the determinant is Sylvester's criterion for positive definiteness. Checking determinant alone would miss the demonstrated two-negative-eigenvalue case. Combined with positive finite lapse/chi and finite full payload, this is a valid saved-metric admissibility check. It neither requires constraints to be small nor asserts perturbation stability, and it cannot certify ghosts between checkpoint times.

## Independent read-only checks

I imported `validate()` with bytecode writing disabled and reran it without writing output files. I independently decoded the header offsets, block strides, logical locations and exponent on these three checkpoints:

| dataset | time/M | ranks / blocks | header bytes | per-block stride | result |
|---|---:|---:|---:|---:|---|
| `evolution/cpu-small-dx025` |244.275|1 /8|17540|1087488|pass,0invalid saved cells|
| `covariant-sources/covariant_const01` |300|1 /1|17760|3663360|fail,8invalid corner ghosts|
| `evolution/gpu-8840650/small_dx0125` |333.3375|8 /8|17543|3663360|pass,0invalid saved cells|

Both eight-block datasets use global-ID locations `(000),(100),(010),(110),(001),(101),(011),(111)` at logical level1, matching the expected Z ordering. Each GPU rank file contains exactly one block.

The failed case's first reported cell is `(-2.875,-2.875,-2.875)M`, ghost depths `(4,4,4)`, with positive lapse5.16582, positive chi3.44754, gxx1.96143 and determinant approximately1, but second leading minor `-1.6702377068`. This confirms the intended stronger metric check detects the invalid saved state despite a nearly unit determinant.

The final exponent guard was then verified in source; the author reran the eight-rank GPU check after adding it. No simulations, restarts, production changes or queue actions were performed for this review.

## Follow-up reconstruction guard and regressions

A second bounded review checked the lapse reconstruction against the constructor defaults and added an explicit rejection when neither `evolve_lapse_residual` nor `preserve_lapse_residual` is enabled. The default chain is analytic background -> evolve gauge -> evolve lapse, with preserve lapse default false. This is necessary because the checker otherwise adds the saved lapse residual while the evolution would ignore it. Existing test configurations evolve the lapse, so no recorded verdict changes. The helper hash above describes the initial review, not this final guarded version; `followup-tests.json` records the final hashes.

Eight synthetic on-disk tests exercise the actual binary parser/cohort and validity checker, including serial/multiple-block and MPI cohorts, positive-determinant indefinite fourth ghosts, NaN fluid, invalid active lapse, default/preserved/unsupported lapse configurations, geometry guards, missing or mismatched ranks and truncated payload. All pass. Independent real checks preserve the sigma1 t300M pass, sigma.1 t300M failure with eight invalid ghosts and eight-rank GPU t333.3375M pass. The parent independently reran all tests and validated the sigma1 final1000M checkpoint with zero invalid active/ghost cells.
