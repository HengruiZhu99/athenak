# Local preflight for the FO-GH/Z4c 32M comparison

Status date: 2026-08-17

This compact record covers campaign plumbing only.  It is not GPU or long-time
qualification evidence.

## Common diagnostic and initial-data checks

- A formulation-independent vacuum ADM kernel now evaluates Hamiltonian and
  momentum from each formulation's reconstructed `gamma_ij` and `K_ij` with
  one selected AthenaK finite-difference stencil.
- Common history output has no lapse or chi mask.  It records proper-volume
  H/M L1 and L2 integrals, Linf, and volume in the fixed regions required by
  the campaign, plus fixed shells around the six refinement interfaces.
- The existing native FO-GH alpha mask and native Z4c chi mask remain separate
  diagnostics.  Native Z4c history literally includes cells with
  `chi >= excise_chi` (default `0.0625`); for the identical initial data
  `chi=alpha^2`, so this is equivalent to `alpha >= 0.25` only initially, not
  necessarily during evolution.
- Serial and two-rank MPI common histories agree to reduction-order roundoff;
  Linf, inverse finest spacing, characteristic speed, and effective CFL use
  MPI maximum reductions rather than sums.
- A uniform-grid FO-GH/Z4c initial-data comparison found maximum absolute
  differences `1.1102230246251565e-15` in common `gamma_ij`, `K_ij`, and
  `psi4`, and `1.1102230246251565e-16` in `alpha` and `beta^i`.
- A two-cycle direct Z4c run and one-cycle checkpoint/restart continuation
  agreed in the final Z4c slice to maximum absolute difference
  `3.7698902159710723e-14`.  The restart log did not re-run analytic puncture
  initialization.

Local builds used GCC 13.3.0 and Kokkos 4.4 Serial.  Both the built-in FO-GH
executable and the custom `z4c/z4c_one_puncture` executable compiled.  The
common-history path also compiled and ran with two MPI ranks.

## One-A100 memory blocker in the controlling grid

AthenaK's authoritative startup tree for the specified root grid, finest
`[-2,2]^3` cube, seven physical refinement levels, and required 2:1 balance is:

```text
Root grid = 4 x 4 x 4 MeshBlocks
Total number of MeshBlocks = 3200
Physical levels 1--6: 448 leaf MeshBlocks each
Physical level 7: 512 leaf MeshBlocks
```

The local N=8 startup reached 25,950,372 KiB resident memory before the
30-GiB host killed it while allocating the FO-GH data.  More decisively, a
lower bound from production arrays alone can be computed without a GPU run.
For each leaf block, the fine arrays `u0`, `u1`, `u_rhs`, native constraints,
ADM adapter, and common constraints contain 224 doubles per allocated cell.
The multilevel coarse FO-GH state adds 63 doubles per coarse allocated cell.
With four ghost cells on every side, the lower bounds are:

| active cells/MB | fine allocation | coarse FO-GH allocation | lower bound |
|---:|---:|---:|---:|
| 8^3 | 21.87 GiB | 2.60 GiB | 24.47 GiB |
| 12^3 | 42.72 GiB | 4.12 GiB | 46.84 GiB |
| 16^3 | 73.83 GiB | 6.15 GiB | 79.98 GiB |

These figures exclude boundary buffers, MeshBlock metadata, Kokkos/runtime
allocations, and output workspaces.  Therefore the medium and fine FO-GH cases
cannot fit on one 40-GB Perlmutter A100.  Even deleting the new diagnostic and
ADM arrays would not make those cases fit: the three 63-field evolution arrays
plus the coarse FO-GH state already require about 40.17 GiB at N=12 and 68.45
GiB at N=16, before boundary buffers.

The prescribed numerical grid is feasible with one MPI rank per A100 on a
four-A100 Perlmutter node (about 800 blocks per rank), but using that resource
requires explicit authorization because the controlling goal says one A100.
