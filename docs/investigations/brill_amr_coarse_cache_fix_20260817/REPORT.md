# Brill Cartoon AMR coarse-cache ownership repair

- Date: 2026-08-17
- Source branch: `codex/brill-amr-coarse-cache-coherence-20260817`
- Repair commit: `ab651f0ebd113f8718fefbf6d802976e6b3e8738`
- Source tree: `fae0f46e52717ab0e9a3f6c3ffc2dbbc0261b96f`

## Verdict

The suspected source defect is real. In Z4c, a same-level neighbor already sends an owner-derived, current-stage `coarse_u0` overlap through `isame_z4c`; `FillCoarseInBndryCC` then recomputed and overwrote part of that received representation with a receiver-local high-order restriction. The two writers agree for polynomial data but do not define one coherent representation for a general field.

The repair establishes one owner-authoritative rule: Z4c preserves the already complete received/local/coarser/physical-boundary cache and skips the redundant same-level local refresh. Generic finite-volume fields retain their old refresh path.

The cache invariant is closed with high confidence. The exact cycle-1722 zero-PDE replay has byte-identical `u0` and `coarse_u0` between the post-BC and nominal refresh phases, no sender/receiver mismatch, no missing production parent, and finite strictly positive chi.

The repair does **not** materially reduce the cycle-1722 production C/H/M/Z jump. Each patched jump factor differs from the old factor by at most about 0.0011%. The cache defect is therefore real but secondary to the dominant constraint injection at this event.

The one matched short N256 continuation also does not improve the production
failure.  It crosses the predecessor's terminal time with essentially the same
state, then continues only because the strict step wall time permits more
refine/derefine churn.  It times out at `t=11.9558403611 M`, with level 19 in
the final history row, `dt=3.5762786865e-8 M`, and `C²=2.3608924968e11`.

No convergence, Figure-3 reproduction, horizon, or production-qualification claim is made.

## Observations

### Exact ownership and consumer inventory

The inventory is derived from the production `buffs_cc.cpp` receive ranges and the full 5x5 O6 prolongation stencil, not from face intuition.

| Quantity | Count |
|---|---:|
| MeshBlocks in accepted new topology | 86 |
| Geometric neighbor relations | 582 |
| Coarser-neighbor relations | 88 |
| Prolongation parent targets | 2,016 |
| Unique coarse cells consumed by O6 | 7,056 |
| Required cells from coarser receive | 4,736 |
| Required cells from local active restriction | 1,624 |
| Required cells from same-level receive | 672 |
| Required cells from physical BC construction | 24 |
| Required cells with no pre-refresh source | 0 |
| Required cells with multiple authoritative sources | 0 |
| Required cells overwritten by local refresh | 344 |

Across the complete topology, all 8,288 locally refreshed cache cells lie inside a 17,920-cell same-level received overlap. The earlier face-only census saw exactly half of 2,688 values change because the receive owns four coarse ghost layers whereas the local refresh reconstructs only the two layers representable by complete local fine pairs.

### Byte-level causal chain on the old captured event

- Current production P5 replay reproduced 201,600 captured fine variable values to maximum absolute error `6.661338147750939e-16`.
- The local refresh changed 6,191 consumed coarse variable values.
- Preserving received values changed 44,261 downstream prolongated fine variable values.
- The nearest changed fine value was `0.11744762795603834 M` from the known seam near `(rho,z)=(5.109375,-0.046875) M`.
- This proves a source-to-cache-to-fine-ghost byte chain. It does not independently prove constraint causality.

### Patched exact zero-PDE replay

- Remote root: `/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-coarse-cache-ab651f0e-v5-20260817`
- Job: `57168348` (`COMPLETED`, exit `0:0`)
- Authenticated restart SHA-256: `83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`
- Executable SHA-256: `2c05dc123811c00c2cb6239e11d4f074bb85e605da467ecc6557f74becd9352f`

The replay advanced the authenticated restart from cycle 1721 to the target transaction at cycle 1722, created 12 blocks, then stopped after T5 and before the next RHS. Direct file comparison shows:

- `t3_02_PHYSICAL_OR_AXIS_BC/coarse_u0.bin` equals `t3_03_SAME_LEVEL_COARSE_REFRESH/coarse_u0.bin` byte for byte;
- the corresponding `u0.bin` files also agree byte for byte;
- `post_receive_same_level_ghost_mismatch=false`;
- `post_refresh_same_level_coarse_cache_mismatch=false`;
- chi and determinant minima are finite and positive;
- production constraints contain no nonfinite value;
- same-rank and MPI2 ownership regressions pass.

### Production constraint jump

Values below are squared proper-volume norms produced by AthenaK C++, using the existing proper axisymmetric ring measure. This is not a fictitious collapsed-y normalization effect.

| Norm² | Old T0 | Old T5 | Old factor | Patched T0 | Patched T5 | Patched factor | Patched/old factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 14.1509602 | 88.2281141 | 6.23477935 | 14.1509665 | 88.2282294 | 6.23478468 | 1.00000086 |
| H | 0.562302591 | 12.8946729 | 22.9319109 | 0.562298777 | 12.8946621 | 22.9320471 | 1.00000594 |
| M | 0.775691622 | 62.5304531 | 80.6125157 | 0.775701684 | 62.5305790 | 80.6116325 | 0.99998904 |
| Z | 2.19527544 | 2.19646472 | 1.00054175 | 2.19527547 | 2.19646475 | 1.00054175 | 1.00000000 |

The invariant correction is numerically immaterial to this transaction's integrated constraint jump.

### Matched short N256 continuation

- Remote root: `/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-coarse-cache-ab651f0e-v6-pde-20260817`
- Job: `57168637`
- Science step: `TIMEOUT`, exit `0:15`, elapsed `5401 s`
- Allocation: `TIMEOUT`, exit `0:0`, elapsed `7229 s`

The run reused the authenticated v5 CUDA executable, source, restart, input,
gauge, damping, O6/RK4/CFL/KO settings, and `dchi_max=0.01`.  It was the only
post-repair PDE run.

| History quantity | Old continuation | Patched continuation |
|---|---:|---:|
| terminal time | 11.9547843933 | 11.9558403611 |
| terminal cycle | 3620 | 4389 |
| last `dt` | 4.57763671875e-6 | 3.57627868652e-8 |
| last C² | 3.42819892964e7 | 2.36089249682e11 |
| last H² | 2.50540612277e7 | 1.62492520451e11 |
| last M² | 9.22787117670e6 | 7.35967290880e10 |
| last Z² | 12.0484588478 | 26.1024925908 |
| last max abs K | 1.89575792055e3 | 1.56928941731e5 |
| last max Kretschmann | 1.49211361079e13 | 1.63227322694e20 |
| last history level / MeshBlocks | 12 / 350 | 19 / 665 |
| maximum history level / MeshBlocks | 13 / 371 | 19 / 698 |

At the first patched row at or beyond the old terminal time
(`t=11.9547866821 M`), C² is `3.4765648102e7`, level is 12, and `dt` is still
`4.57763671875e-6 M`: the onset is materially the same.  Thereafter repeated
refinement and derefinement drives the patched run to much smaller timesteps
and much larger constraints.  The final log records a level-19-to-20 creation
transaction at cycle 4390 immediately before the step timeout.

This is evidence that preserving owner-derived `coarse_u0` does not stabilize
the short evolution.  It is not evidence that the repair makes the physics
worse: the patched job was allowed to evolve the already-runaway state much
longer in cycle count than the predecessor.

## Deductions

1. The same-level received cache is a complete current-stage source for every Z4c O6 parent needed in this topology; the local refresh has no legitimate missing-corner role here.
2. Multiple writer semantics were incorrect even though eliminating them does not cure the event jump.
3. The dominant cycle-1722 constraint injection originates elsewhere in the regrid/transfer transaction. Leading bounded candidates include the newly created active representation, derivative reconstruction across the new hierarchy, and parent under-resolution. This report does not select among them.

## Smallest recommended next diagnostic

Repeat only the already authenticated cycle-1722 zero-PDE transaction with a
production-C++ constraint census on the same accepted leaf ownership at two
additional boundaries:

1. immediately after newly created active cells are populated, before any
   ghost reconstruction; and
2. immediately after the complete ghost/BC/prolongation sequence, before
   algebraic projection.

For every changed cell, record distance to the nearest coarse-fine interface,
axis status, parent stencil, the production O6 value, and a shadow O4 value.
The decision rule is bounded: a jump already present at (1) selects active
transfer/parent under-resolution; a jump introduced only at (2) selects
derivative or ghost reconstruction.  If neither changes while the final
history norm does, audit the constraint recomputation itself.  Do not change
the transfer, gate, gauge, damping, or AMR threshold in this diagnostic.

## Hypotheses requiring further evidence

- The old overwrite may contribute small local seam error or longer-term phase error even though its event-integrated norm effect is negligible.
- The dominant event jump may be inherent to transferring an under-resolved parent field into a higher-order child representation.
- A separate high-order restriction/prolongation edge closure may dominate the injected constraint signal.

## Unsupported possibilities

- It is unsupported to call this a convergence result or Figure-3 reproduction.
- It is unsupported to attribute the full Brill instability to this cache bug.
- The unqualified independent Python ADM-constraint port is not used for causal claims.
- P8, chi floors/clipping, weaker positivity gates, gauge changes, damping changes, or parameter sweeps are outside scope.

## Regression coverage

| Dimension/topology | NGHOST | Orientation | Ownership |
|---|---|---|---|
| Cartoon 2D off-axis and axis-adjacent mixed-level T-junction | 2, 3, 4 | all four faces; low/high children | same-rank and MPI2 |
| Cartesian 3D mixed-level topology | 2, 3, 4 | representative face/child cases | same-rank and MPI2 |

All 25 Z4c components use a smooth non-polynomial state; chi is finite and strictly positive. Tests verify fine ghosts, received coarse overlap, complete parent sourcing, no later overwrite, positive parents, and identical same-rank/MPI prolongation. Generic non-Z4c refresh remains enabled.

## Source change

- Add a small ownership policy, `ShouldLocallyRefreshSameLevelCoarseCache`.
- Return before the same-level local refresh kernel only for Z4c.
- Preserve the pre-existing generic finite-volume path.
- Add a C++ ownership regression, MPI2 variant, static routing checks, and an exact captured-event ownership/replay audit.

No transfer coefficient, prolongation order, chi policy, PDE, gauge, damping, AMR threshold, CFL, or KO setting changes.

## Evidence map

- `coarse_cache_ownership_summary.json`: exact writer/consumer and byte-replay summary.
- `coarse_cache_ownership_cells.csv`: each consumed coarse cell and writer classification.
- `changed_consumed_coarse_values.csv`: every overwritten consumed variable value.
- `preserve_received_fine_differences.csv`: every altered downstream fine value.
- `coarse_cache_overwrite_map.png`: physical location of overwritten consumed cache data.
- `preserve_received_fine_difference_map.png`: physical location of downstream differences.
- `zero_pde_comparison.json` and `zero_pde_constraint_jump_comparison.png`: authoritative C++ old/patched event comparison.
- `short_pde_comparison.json` and `short_pde_history_comparison.png`: matched old/patched history and terminal comparison.
- `remote_v5_zero_pde/`: selected remote logs, manifests, aggregates, and production-census outputs. The complete 489 MiB raw event remains immutable at the remote v5 root and is hash-listed by its run manifest.
- `remote_v6_short_pde/`: selected run history, log, command, binding,
  environment, source-status, accounting, and complete 3,247-entry raw-run
  manifest.  The raw 83 GiB tree remains on Perlmutter.

## Integrity and limitations

- v5 zero-PDE run manifest SHA-256:
  `bba4126d3d0f33f4d63369feb3bb8894abeb21255ea3a4a41ebea84d87b5d40c`;
  detached-file SHA-256:
  `74bbb861b5d0aa5c479d54887fd34c032456f0c683209661b83c539335a429a5`.
- v6 short-PDE run manifest SHA-256:
  `d2dce5a4fccdfb1a97507256de5dcb3765aa60d771d988d53a683854ace012e4`.
  Its detached file contains the same digest.  The allocation manifest is
  `200b5d6372cf40f081aa74f3593be5e9e20bd6b92f89f8e95ba6f9c306252c70`.
- v6 run-log SHA-256:
  `3044d88330d1130aad2c97f2fce66976749509a77d698d60325be6744f2cc958`;
  history SHA-256:
  `8836677539f12608a6a385d3d0ff571f68e5a28f3511210809eee195b550beba`.
- The science step timed out first.  The allocation later reached its outer
  wall limit while finalizing, after both complete run-manifest files had been
  written.  The allocation-side manifest and settled accounting also verify.
- No convergence, Figure-3 reproduction, horizon, or production qualification
  follows from these results.  The data isolate one cache-writer defect and
  show that it is secondary; they do not yet isolate the dominant constraint
  source.

Harness-only predecessor failures v1-v4 are preserved remotely and are not scientific results.
