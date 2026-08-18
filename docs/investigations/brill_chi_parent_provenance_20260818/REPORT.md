# N256 replay chi-parent provenance diagnostic

Date: 2026-08-18

Final disposition: `ACTIVE_FINE_CHI_FAILURE`

## Result

The first loss of the strict finite-positive chi invariant occurs in the active
fine-grid state immediately after `ExpRKUpdate`, before `RestrictU` or any
coarse-cache communication, physical/axis ghost filling, or parent-stencil
validation.

The first transition is at cycle 5546, RK stage 3, coordinate time
`0x1.5124ccccccd9bp+3` (10.535742187500366 M), on replay tree checksum
`cf359244e1483352`. Two active cells become negative:

| GID | logical location | local (j,i) | rho | z | chi |
|---:|---|---|---:|---:|---:|
| 35 | level 5, (20,63,0) | (67,39) | 5.138671875 | -0.001953125 | -1.3316138503433481 |
| 60 | level 5, (20,64,0) | (4,39) | 5.138671875 | +0.001953125 | -1.3409347716159825 |

These are an equatorially paired failure at the active edges of adjacent
same-level blocks. At stage 2 of the same cycle, the global active minimum was
still 0.35710902688047425 and the consumed coarse minimum was also positive.
No nonfinite value was observed.

This result activates the goal's mandatory stop condition. No restriction
fallback, cache/ghost correction, corrected N128 authority, N256 replay
control, or native-N256 control was implemented or run.

## What is established

- High-order restriction is not the first writer of the terminal invalid chi.
- Communication, axis/physical-boundary filling, same-level coarse refresh and
  the P5 parent gate occur downstream of the first invalid active value.
- The earlier terminal report of 3928 rejected parent stencils was therefore a
  downstream consequence for this replay, not evidence that restriction first
  created the bad state.
- The two cells are close to the equatorial plane but not on the Cartoon
  symmetry axis; their paired locations are consistent with equatorial
  symmetry. This does not identify the responsible RHS term.
- Replay authority events 10 and 11 were applied at their exact hexadecimal
  physical times with zero ULP difference. Event 11 used a clipped preceding
  timestep as designed. The accepted trees contained 86 and 98 leaves,
  respectively, and matched exactly.
- The shadow N256 criterion emitted 63,002 requests from cycles 4097--5546:
  62,312 derefinement and 690 refinement requests. The final two failing blocks
  had requested refinement by cycle 5546, but the active-state invariant was
  already lost at S0. Consequently this diagnostic does not establish
  `REPLAY_TREE_UNDERREFINES_N256` as the primary failure classification.

## Observation versus inference

Observation: chi is positive through stage 2 S0--S4 at cycle 5546, then two
active cells are negative at stage 3 S0 immediately after the RK update.

Inference: the upstream mechanism lies in the PDE/RK update or data consumed by
that update, rather than in the subsequent restriction/cache/BC/prolongation
pipeline.

Unresolved hypothesis: one or more chi RHS contributions may become stiff or
under-resolved at the adjacent-block/equatorial feature near rho=5.1387. The
present evidence does not distinguish a bulk formulation/gauge instability,
an interface-contaminated RHS stencil, or a timestep/stage stability failure.

## Run and provenance

- Dedicated allocation: job `57214220`, QOS `gpu_shared_interactive`, one GPU,
  node `nid008257`. The unrelated pre-existing interactive job was not used or
  modified.
- The allocation was released after evidence capture. The parent accounting
  state is therefore `CANCELLED by user`; terminal diagnostic step `.7`
  completed successfully.
- Source HEAD: `ac75c8d348da91b38cbc6855b5fba51cd3089663`
- Source tree: `6284882bd06e8db379495675aba7a4f153fb4afa`
- Kokkos: `6739bc623081648af9e752b616d9671527922cbf`
- Diagnostic executable SHA256:
  `b4a5681c23310af12f2192faadc78a01fb12a1713fc6ab7707217be637d1a746`
- Authenticated restart SHA256:
  `2e2e8f7febd0d4fbb204f172df149f9295de6aa66097ef3c9f19048aa29a20e9`
- Remote evidence root:
  `/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-parent-provenance-ac75c8d3-v1-20260818`

The default-off diagnostic was intentionally left uncommitted. Its exact
tracked patch, owned-source tar, status, HEAD/tree/Kokkos identity and hashes
are under `evidence/final-diagnostic-source/`.

Five focused CUDA/MPI tests passed: chi prolongation, coarse-cache ownership,
AMR-history format, derefine-factor static, and Cartoon-AMR static.

## Harness lineage

Several harness-only attempts are preserved in accounting and the remote root:

- `.1`: combined restart/input parser misuse; no PDE result.
- `.3`: CUDA host-copy layout failure; no RK advance.
- `.4`: output-directory collision; no RK advance.
- `.5`: intentionally cancelled while optimizing diagnostic copying; no
  scientific conclusion.
- `.6`: full diagnostic reached the scientific stop, but a downstream harness
  check still expected coarse-failure files and returned 1.
- `.7`: strict terminal rerun captured the two active cells and completed 0.

## Intentionally absent conditional outputs

`first_invalid_coarse_cell.json`, `first_invalid_parent_stencil.csv`, unique
invalid-coarse-cell tables, restriction-candidate comparisons and bounded
corrected-run summaries were not produced. This is the correct consequence of
stopping at an earlier `ACTIVE_FINE_CHI_FAILURE`: no invalid coarse cell or
conditional production correction was reached.

## Smallest justified next diagnostic

Replay only the same bounded terminal window and instrument the two identified
cells at cycle 5546 stage 3. Record the pre-update chi, complete RHS, RK
coefficient/update increment, and a term-by-term chi RHS decomposition, plus
the exact derivative stencil values and provenance used by the RHS. Stop at
the first negative candidate. This can distinguish an already-corrupt RHS
stencil from a locally excessive but finite update without changing gauge,
KO, CFL, damping, AMR transfer, or positivity gates.

No convergence, Figure-3 reproduction, continuum-instability, or production
qualification claim is made.
