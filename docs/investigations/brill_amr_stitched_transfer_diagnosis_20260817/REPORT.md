# Brill N256 cycle-1722 zero-PDE AMR seam diagnosis

## Verdict

The bounded diagnostic classifies the event as
`concrete_ghost_or_cache_bug_isolated`, with `qualification_claim=false`.

The ordinary same-level Z4c ghost exchange is not the failure found here.  All
5,632 checked fine-grid face ghost cells agree exactly with the corresponding
sender active cells after receive and through the later boundary phases.  The
first loss of the tested same-level representation invariant occurs in
`FillCoarseInBndryCC`: the coarse-cache receive state is exact for all 2,688
checked face cells at `T3_01_MPI_RECEIVE`, then
`T3_03_SAME_LEVEL_COARSE_REFRESH` changes 1,344 of them above the
roundoff-scaled gate.  The largest relative and absolute discrepancy is
0.28677378470336334 in `Khat`, near the previously identified artificial seam
at approximately `(rho,z)=(5.109375,-0.046875) M`.

This isolates a coarse-cache coherence/seam-semantics problem, not an axis
parity-ghost error and not a fictitious collapsed-y history normalization.
It does not by itself prove that this one defect explains the complete late-time
runaway.

## Provenance and scope

- Repository: `https://github.com/HengruiZhu99/athenak`
- Branch: `codex/brill-amr-frozen-hierarchy-20260816`
- Captured production source: `b465629caf20be81c4f19f7c818fd3e0b9b2c242`
- Captured production tree: `9f797582eb44b9dc079457529e50474a7dd129d5`
- Declared restart SHA256:
  `83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`
- Target: cycle 1722, `t=9.50625 M`, 74 to 86 MeshBlocks
- Selected refined parents: GIDs 28, 29, 45, 48
- Selected refined children: GIDs 28-35, 51-54, 57-60
- Analysis type: offline and zero-PDE; no new evolution was launched

The local raw event root is recorded in `provenance.json`.  Its T1 phase JSON
has SHA256 `18a7ba6b3012658c435d3a4a73cf43a7bf851beacfdaba63cd493edf7e50da06`.
The restart hash is inherited from the sealed provenance; the restart byte was
not redundantly copied into this compact Git handoff.

## OBSERVATIONS

### Production reconstruction and stitched support

- The offline `P5_BLOCK` reconstruction reproduces every production T2 child
  active value.  The maximum absolute difference over all 16 children is
  `1.7763568394002505e-15`.
- The algebraic projection reproduces production T4 on the common supported
  lattice to `4.440892098500626e-16`.
- The stitched source contains no duplicate parent centers.  Both L4-to-L5 and
  L5-to-L6 comparisons retain 6,222 child cells whose complete interpolation
  and O6 derivative stencils are supported by genuine same-level parent active
  data.
- Raw captured `C`, `M`, and `Z` are finite and nonnegative at T0 and T5.

### Fine-grid same-level ghost census

For every checked phase from T0 through the final post-prolongation physical
boundary fill, the maximum sender-active versus receiver-ghost discrepancy is
exactly zero.  The post-receive phases each cover 5,632 face ghost cells, all 25
evolved variables, ghost depths 1 through 4, and the selected target
neighborhood.

### Same-level coarse-cache census

| phase | cells | max abs/relative discrepancy | above gate |
|---|---:|---:|---:|
| `T3_00_RESTRICT` | 2,688 | 0.12350358680721751 | 2,688 |
| `T3_01_MPI_RECEIVE` | 2,688 | 0 | 0 |
| `T3_03_SAME_LEVEL_COARSE_REFRESH` | 2,688 | 0.28677378470336334 | 1,344 |

The receive operation establishes an exact common value.  The subsequent
refresh is therefore the first writer that breaks the tested coherence
invariant.  The source path is:

1. `Z4c::Prolongate` calls `FillCoarseInBndryCC` in
   `src/z4c/z4c_tasks.cpp`;
2. `MeshBoundaryValuesCC::FillCoarseInBndryCC` launches `ProlCCSame` in
   `src/bvals/prolongation.cpp`;
3. the 2D Z4c O6 branch writes `coarse_u0` through
   `RestrictInterpolation<4>`;
4. the restriction stencil and edge orientation are selected from receiver-
   local fine indices in `src/mesh/restriction.hpp`.

The sender-owned coarse value and receiver-local recomputation use different
block-local stencils for the same physical coarse location on non-polynomial
data.  The refresh overwrites the exact received representation with the latter.

### P5/P8 diagnostic comparison

`P5_BLOCK` and `P5_STITCHED` evolved active values and their derivative
diagnostics agree to roundoff on the common lattice.  Thus merely stitching the
parent active lattice does not remove the event's main L5-to-L6 error.

The diagnostic P8 reconstruction reduces the exploratory L5-to-L6 proper-ring
integrals relative to P5_STITCHED to approximately 0.345 for C, 0.415 for H,
and 0.330 for M.  It does not provide comparable factor-two improvement in the
L4-to-L5 balance region.  This is mixed evidence for interpolation-order
sensitivity, not a production recommendation.

### Constraint reconstruction qualification

The P5 production constraint reference in the tables is the captured
production constraint state.  The independent NumPy port of the O6 Cartoon ADM
constraint operator does **not** satisfy its strict reproduction gate; see
`production_reproduction_validation.json`.  Therefore stitched P5/P8
constraint ratios are explicitly marked `exploratory_only` and are not used to
select the disposition.  The disposition rests on direct captured byte
comparisons in the ghost/coarse-cache census.

The Cartoon history measure is already the proper axisymmetric ring measure,
`2*pi*rho*sqrt(gamma)*drho*dz`.  No fictitious collapsed-y cell width is present.

## DEDUCTIONS

1. The suspected failure is not a general failure to update ordinary fine-grid
   same-level ghosts: those bytes match their senders exactly.
2. The same-level coarse cache is coherent immediately after receive and is
   made incoherent by the local coarse refresh.  This establishes the first
   incorrect writer within the captured transaction.
3. The problem is off-axis and MeshBlock-seam correlated.  It is not isolated
   to the symmetry-axis parity boundary.
4. Because the cache values feed coarse-to-fine prolongation corners during
   ordinary multilevel evolution, repeated regridding and repeated boundary
   reconstruction can repeatedly expose this representation mismatch.

## HYPOTHESES

The narrow leading hypothesis is that `FillCoarseInBndryCC` should not
overwrite a sender-authoritative same-level coarse value with a different
receiver-local high-order restriction stencil.  A production correction should
either preserve the received value for those locations or define one canonical
global stencil/orientation and use it on both sides.

The smallest regression should build two same-level 2D Cartoon blocks adjacent
to a coarse neighbor, fill a non-polynomial but smooth 25-component state,
perform receive then same-level coarse refresh, and require:

- received same-level coarse overlaps remain equal to sender-authoritative
  values to roundoff;
- only the genuinely required coarse/fine corner targets are filled;
- NGHOST 2, 3, and 4 paths are covered;
- axis-adjacent and off-axis arrangements are both covered;
- the ensuing positive-chi prolongation sees identical parent stencils on both
  ownership routes.

No production change was made in this handoff.

## UNSUPPORTED POSSIBILITIES

- This does not establish convergence or reproduce Figure 3.
- It does not show that the symmetry-axis ghost fill is wrong.
- It does not prove that P8 should become the production transfer.
- It does not distinguish all later-time bulk, gauge, or under-resolution
  mechanisms.
- It does not justify a chi floor, weakened positivity gate, parameter sweep,
  or relaxed tolerance.
- It does not qualify the exploratory stitched constraint ratios because the
  independent Python constraint port missed its strict production gate.

## Natural next step

Implement only the focused two-block-plus-coarse-neighbor regression described
above.  Use it to decide whether preserving sender-authoritative coarse overlap
values or canonicalizing the restriction stencil is the minimal correction.
After that source-level invariant closes, rerun only the same zero-PDE cycle-1722
event before considering any evolution.

## Artifact guide

- `verdict.json`: strict disposition and qualification boundary
- `provenance.json`: source, event, restart, and raw squared-field checks
- `topology_stitching_manifest.json`: selected topology and P5 validation
- `same_level_ghost_summary.csv`: fine-grid ghost census
- `coarse_cache_summary.csv`: first-writer cache census
- `coarse_cache_worst.csv`: worst cells and coordinates
- `production_reproduction_validation.json`: projection and constraint-port audit
- `constraint_metrics.csv`: production P5 and exploratory stitched metrics
- `representation_errors.csv`, `derivative_disagreements.csv`,
  `high_frequency_diagnostics.csv`: state/operator diagnostics
- `*_stitched_comparison.png`: maps on the common valid lattice
- `REMOTE_REVIEW_PROMPT.md`: bounded external-review request
