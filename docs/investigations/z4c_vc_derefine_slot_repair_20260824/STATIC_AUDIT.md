# Native-VC derefinement slot-lifecycle static audit

Date: 2026-08-23

Repository: `https://github.com/HengruiZhu99/athenak`

Base branch: `codex/z4c-vc-figure3-convergence-20260823`

Exact base commit: `3c240a370084b0d2b749932386cf057921b575ef`

Base tree: `bba0717be9f4b8b34ed84c593256ad44f4b22ce3`

Working branch: `codex/z4c-vc-derefine-slot-repair-20260824`

Kokkos: `6739bc623081648af9e752b616d9671527922cbf` (`4.7.02`)

## Disposition

```text
EVENT3_MAP_MATCHES_HYPOTHESIS
```

The event-3 map was reconstructed independently from the authenticated AMR
history and the production tree/map algorithms. It matches the proposed map.
The unpatched same-rank native-VC path writes the second reconstructed parent
into slot 26 during A5, although slot 26 still contains the source for the
unaffected block that must move to slot 23 during A6. A6 consequently copies
the wrong parent to block 23 and then replaces the correct parent at slot 26
with the old lower-child array from slot 29.

This is a static proof of a slot-lifecycle violation. The required executable
red test remains the next gate; no production source was changed in this
phase.

## Source and evidence authority

The evidence branch has two documentation commits above the compiled source
commit `6ad9cf4048af6a93aa73cf9940fc78c3b439c8fe`. A path-limited diff over
`src`, `tst`, and `CMakeLists.txt` is empty, so the branch-head mesh/VC source
is identical to the documented production source.

The authority file is:

```text
docs/investigations/z4c_vc_figure3_convergence_20260823/
  evidence/authority/n256_amr_history.jsonl
```

Its SHA-256 is recorded in `EVENT3_MAP.json`. The N256 command used exactly
one MPI rank and one A100 GPU, so for the event-3 transaction both
`first_old` and `first_new` are zero.

One serialization detail matters. `AMRHistory::CurrentLeaves` sorts logical
locations before writing JSON, whereas `MeshBlockTree::CreateZOrderedLLList`
walks the tree recursively in logical-child order. Treating the JSON array
index as a GID gives a false map. `derive_event3_map.py` rebuilds the quadtree
from the sorted leaf set and then performs the same child-order traversal as
`CreateZOrderedLLList`.

## Independent derivation

Event 2 contains 50 leaves. Event 3 contains 44 leaves and replaces these two
four-child families:

| Parent | Old children in Z order | New parent GID |
|---|---|---:|
| `L4 (2,7,0)` | 16 `L5 (4,14,0)`, 17 `L5 (5,14,0)`, 18 `L5 (4,15,0)`, 19 `L5 (5,15,0)` | 16 |
| `L4 (2,8,0)` | 29 `L5 (4,16,0)`, 30 `L5 (5,16,0)`, 31 `L5 (4,17,0)`, 32 `L5 (5,17,0)` | 26 |

`MeshBlockTree::Derefine` assigns each new parent the GID of its lower child.
`CreateZOrderedLLList` then emits the new leaf sequence and writes that
inherited old GID into `newtoold`. Applying the production `oldtonew`
construction gives the following exact portion of the map:

| Old GID | Old logical location | `oldtonew` | New logical location |
|---:|---|---:|---|
| 14 | `L4 (2,6,0)` | 14 | `L4 (2,6,0)` |
| 15 | `L4 (3,6,0)` | 15 | `L4 (3,6,0)` |
| 16 | `L5 (4,14,0)` | 16 | `L4 (2,7,0)` |
| 17 | `L5 (5,14,0)` | 16 | `L4 (2,7,0)` |
| 18 | `L5 (4,15,0)` | 16 | `L4 (2,7,0)` |
| 19 | `L5 (5,15,0)` | 16 | `L4 (2,7,0)` |
| 20 | `L4 (3,7,0)` | 17 | `L4 (3,7,0)` |
| 21 | `L3 (2,2,0)` | 18 | `L3 (2,2,0)` |
| 22 | `L3 (3,2,0)` | 19 | `L3 (3,2,0)` |
| 23 | `L3 (2,3,0)` | 20 | `L3 (2,3,0)` |
| 24 | `L3 (3,3,0)` | 21 | `L3 (3,3,0)` |
| 25 | `L4 (0,8,0)` | 22 | `L4 (0,8,0)` |
| 26 | `L4 (1,8,0)` | 23 | `L4 (1,8,0)` |
| 27 | `L4 (0,9,0)` | 24 | `L4 (0,9,0)` |
| 28 | `L4 (1,9,0)` | 25 | `L4 (1,9,0)` |
| 29 | `L5 (4,16,0)` | 26 | `L4 (2,8,0)` |
| 30 | `L5 (5,16,0)` | 26 | `L4 (2,8,0)` |
| 31 | `L5 (4,17,0)` | 26 | `L4 (2,8,0)` |
| 32 | `L5 (5,17,0)` | 26 | `L4 (2,8,0)` |
| 33 | `L4 (3,8,0)` | 27 | `L4 (3,8,0)` |
| 34 | `L4 (2,9,0)` | 28 | `L4 (2,9,0)` |
| 35 | `L4 (3,9,0)` | 29 | `L4 (3,9,0)` |

The corresponding `newtoold` segment is:

```text
new 14..29 -> old 14,15,16,20,21,22,23,24,25,26,27,28,29,33,34,35
new 30..35 -> old 36,37,38,39,40,41
```

The complete old/new sequences and maps are machine-readable in
`EVENT3_MAP.json`.

## A4/A5/A6 writer order

A4 packs any MPI transfers before local arrays are modified. It does not alter
the local one-rank event-3 state.

At A5, `DerefineVCSameRank` currently computes both `source_base` and
`destination_m`, but writes the active parent to `destination_m`:

```text
family 1: source_base=16, destination_m=16
          u0[16] <- exact-injection parent from coarse_u0[16:20]

family 2: source_base=29, destination_m=26
          u0[26] <- exact-injection parent from coarse_u0[29:33]
```

The first family is safe by coincidence because its old lower-child staging
slot and final new slot are both 16. The second family moves left by three
slots. Its A5 destination is not staging storage: old slot 26 is still the
only source for the unaffected `L4 (1,8,0)` block.

At A6, `CopyVC` delegates to `CopyCC`, which deliberately interprets sources
in the old slot layout and processes left moves in increasing old-GID order:

```text
old 26 -> new 23: u0[23] <- u0[26]   # reads the wrong parent written at A5
old 29 -> new 26: u0[26] <- u0[29]   # replaces the correct parent by lower child
```

Thus two final logical blocks are wrong. The unaffected block at new 23
contains the second parent, while the parent at new 26 is a full-array copy of
its old lower child instead of the reconstructed exact-injection parent.

## Why existing tests and diagnostics did not exclude it

The deterministic dynamic-AMR fixture derefines only one family. Its parent
does not move relative to its old lower-child slot, so `destination_m ==
source_base`; it cannot exercise the overwrite.

The nonconstant linear-wave test likewise uses a single target family. It
checks end-to-end error but does not retain a pre-A5 logical-block oracle for
multiple families in one transaction.

Exact shared-node spreads also do not exclude this bug. Both bad operations
copy whole, internally consistent arrays. Neighboring duplicates can agree
bitwise even while an entire block is registered to the wrong logical
location.

## Same-rank and split-rank scope

The event-3 defect is same-rank and occurs before MPI unpack. The narrow repair
must stage each reconstructed parent in `source_base`, matching
`DerefineCCSameRank`, and let A6 relocate it.

The split-rank path is a separate defect:

- `DerefineVCSameRank` skips the whole family if any sibling is remote;
- receive metadata is created per selected child;
- each VC receive independently writes its quadrant directly into the same
  parent active array;
- quadrant ranges include coincident midpoint planes and the center, so
  multiple receive teams can overlap;
- local child contributions are not deterministically assembled when the
  family is split.

That path did not execute in the one-rank authority event and must have its own
red tests and commit after the same-rank repair.

## Audited source identities

| Path | Git blob | SHA-256 |
|---|---|---|
| `src/mesh/mesh_refinement.cpp` | `96e208021bbff47963ba4eaf9400820ccf969965` | `fd1b6f64669ec2c41296410e214491a1fd7ea8361ba6bd3aa6a062f2fc465d1b` |
| `src/mesh/mesh_refinement_vc.cpp` | `7c4f2803575214ba8b37b19ec87187a56a8d56c2` | `a3185633a893514f80f1cf718487d565bc9cb75f901332cfad29a1104aaa176f` |
| `src/mesh/load_balance.cpp` | `d4c4621f630566a858ce7f618a305dbec25b5e0c` | `1dd4500fc1a9f2e138861beffa127ff1f66cf6ac305a0578cb5e39f61523a2b4` |
| `src/mesh/meshblock_tree.cpp` | `9d46807126cf6507fa86632c5ed16db885409858` | `85deb74719bdcca58968c8d13575c3a586ab92afdac87fc1dd4babe06ff5ee99` |
| `src/mesh/vertex_amr.hpp` | `3baed5dbb58b13666796f3db6f9db548c2b9a6bd` | `93e55ed08490f8aac4c26ec167628ab3678e58a7a61e644303016aa72e26004a` |
| `src/bvals/bvals_vc.cpp` | `03f05fd66f6be06c1af7581096cd73cc095c1d68` | `73085fa0257f76b9eaa58463aea131bedb7fb020d69cef45617e55f1882d7aa1` |
| `src/z4c/z4c_tasks.cpp` | `0dfc5dc86523c4266d9a76f9802b6ef92b33ca13` | `3d65533bd627dc60e64b78291694e64d557154b9f537a6108d6eafc26fd4c7e1` |
| `src/z4c/z4c_vertex_topology.cpp` | `90e768740fb038074079d557fc65ea48ce8f812b` | `229bf67649e16cbb61e393807acafbac0e84d52902983e5d9ffe04990abef377` |
| `tst/unit/z4c/z4c_vertex_dynamic_amr_test.py` | `b904b30320105bc7c8a8dc3590fb4d45e3420f64` | `530a36ee7aeafad5e307f06d8e8879307dd8fecaf2623057a86e45ad8ea72226` |
| `tst/unit/z4c/z4c_vertex_dynamic_linear_wave_test.py` | `b2aa597b535e378225261c622a172f37830b2c51` | `32d387043b8467a826d2455200154d951d7a7f1215eaad34a843b224eff0a0be` |

## Toolchain record at audit start

```text
cmake 3.28.3
GNU C++ 13.3.0
GNU MPI wrapper C++ 13.3.0
host logical CPUs: 32
```

The planned first executable gate is an MPI-enabled Release OpenMP build using
the pinned Kokkos submodule. Exact configure, build, and test commands and their
return codes will be recorded with the red-test evidence rather than claimed
before execution.
