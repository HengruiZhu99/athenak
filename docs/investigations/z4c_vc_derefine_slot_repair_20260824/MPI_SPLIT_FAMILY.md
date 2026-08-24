# Split-rank native-VC derefinement

## Red result

The bounded Perlmutter discriminator used three 2D sibling families and an
equal-cost partition. The middle family has old children 18:21 and new parent
15. With two ranks, the lower child and target parent are on rank 0 while the
upper siblings cross to rank 1. This is the required layout B.

On commit `941a386f090ff6f04910198ea24bb3f9ee2f12af`, the one- and two-rank
executions completed the same physical hierarchy and event times but their
post-derefinement fields differed at 3,173 binary-output values. The maximum
absolute difference was `2.3450449225492775e-04` and the RMS difference was
`6.495238513420746e-06`. Parent GID 15 contained 2,565 of the differing
values; four neighboring blocks differed after boundary reconstruction.

The first recorded mismatch was `chi` at parent-adjacent stored index
`[gid=8,k=0,j=16,i=0]`: one-rank `1.0000348091125488`, two-rank
`1.0000083446502686`. The rank-canonical payload hashes were:

```text
one rank  0fb5b6c8eb5402a3d8b0ef9a179e89f061e573ce13612541651f0be72a83607c
two ranks 7ababd1b72a684a530bf3a7e441ae7da32eb992c894f72107462dff238cd3c50
```

Perlmutter job `57499665` completed in eight seconds. The CPU/OpenMP MPI
executable SHA-256 was
`e2586a6b04d5db21050ac3f5b6f9e3134561deaa9bccd92142631ad87f45c3dc`.

Observation: the target parent is rank dependent. Source audit shows that
`DerefineVCSameRank` skips the complete family when any sibling is remote,
while `InitRecvAMR` receives only the remote upper siblings when the lower
child and parent remain local. A6 therefore copies the lower child's full
active array into the parent, and A8 overwrites only the received quadrants.

## Green result

Commit `661d25d67451f7a920709efe8b065d0a08028f6c` replaced independent
receive-buffer writers with one parent-variable writer. Children are visited in
canonical logical-child order, coincident values are checked before one
canonical value is written, and local and received siblings each contribute
exactly once. The ordinary CC/FC unpack paths are unchanged.

The first Python harness attempt was invalid because the binary reader returns a
list of per-block arrays; commit `9247a595feb3cd8d2485a23d0e72d5ac3e9c02f5`
stacks those arrays before comparison. This is a test-only correction. On that
revision the pure-derefinement MPI2 and MPI4 cases were bitwise identical to the
one-rank reference for all 25 variables (Perlmutter job `57499971`).

## Mixed refine/derefine red result

The pure-derefinement result did not cover A7. A bounded mixed transaction was
therefore added in commits `a9ff43eb4e88648ef750920543b5ae2d71f3d2d7` and
`8ce7b3002d0cbd8760649cf284b7a96646c1c34b`. It derefines the same three
families while refining root Z-order GID 16. Its new child slots overlap old
local child slots 18:20 of the split family rooted at GID 15.

On job `57500192`, MPI2 passed but MPI4 failed at A8 with:

```text
native VC split-derefinement assembly
missing_or_nonfinite=0 inconsistent_shared_vertices=323
```

This is direct evidence that A7 `CopyForRefinementVC` overwrote at least one
local coarse-child contribution before the A8 split-parent writer consumed it.
It is not a tolerance or physics failure: the same mixed one-rank transaction
passed, the failure occurred before evolution resumed, and the existing strict
shared-vertex consistency check detected it.

## Mixed-transaction repair and qualification

Commit `8548bc2ffe5a7eef558125c9a472e5ec61504e53` constructs the split-parent
source map at A3 while the old hierarchy is authoritative and snapshots only
the required local child coarse arrays into compact immutable storage. A8 then
assembles each parent from those snapshots and the completed receive buffers.
The snapshot is explicitly fenced before A4/A5/A6/A7 can reuse old storage.

Perlmutter job `57500409` ran the complete focused matrix against executable
SHA-256
`ded6117c9532f042a3c720e7e560c1a02472d3cc4024ef99d959c31055b1b449`:

| Test | Result |
|---|---|
| pure derefinement, MPI2 | bitwise pass, all 25 variables |
| mixed refine/derefine, MPI2 | bitwise pass, all 25 variables |
| pure derefinement, MPI4 | bitwise pass, all 25 variables |
| mixed refine/derefine, MPI4 | bitwise pass, all 25 variables |

The allocation completed normally in 15 seconds. The CPU/OpenMP executable was
run with `MPICH_GPU_SUPPORT_ENABLED=0`; the reserved A100 was not used by this
CPU-backend test. The corresponding same-rank host matrix remains green at
8/8, including 2D/3D, O2/q4, O4/q6, O6/q8, constant, smooth nonconstant, and
mixed refine/derefine cases.

These results qualify the exercised ownership layouts and the A7 overlap
discriminator. They do not yet replace the required authority event-3 replay,
CUDA checks, or early three-resolution convergence gate.

## Completed ownership matrix at final source

Commits `65af69734798b91d6e2fdf0494393fec67f0fc45` and
`6dd20656a305f2543bbbd7001550c6ac67019180` complete the previously missing
ownership discriminators. The final matrix is:

| Required layout | Fixture | MPI ranks | Evidence |
|---|---|---:|---|
| A: siblings and target local | local families within every comparison | 2, 4 | bitwise match |
| B: local lower child/target, remote upper sibling | `local_lower` | 2, 4 | bitwise match |
| C: lower child remote from target; another sibling local | `remote_lower` | 2, 4 | bitwise match; move right |
| D: split family plus unrelated migration | `local_lower --mixed-refine` | 2, 4 | bitwise match |
| E: two split families target one rank | `dual_split` | 4 | bitwise match |

Layout E is a four-rank partition discriminator by construction: old families
8--11 and 21--24 cross different old partitions, while new parents 8 and 12
both target rank 1. Every multi-rank run is compared with its one-rank layout-A
reference at the same accepted time and hierarchy. The comparison is bitwise
for all 25 fields and includes physical and logical MeshBlock metadata.

Perlmutter job `57504956` ran the final-source CPU dual-split case and all
CUDA MPI2/MPI4 ownership cases. Every case returned zero. The bundle is
`evidence/regressions/perlmutter/regressions_6dd_targeted_v1`; its
`SHA256SUMS` file hashes to
`e655a842435ecf532e2b31901f3b3d91cc75fbccb4cb5d1a72f1ac9b430e999f`.
