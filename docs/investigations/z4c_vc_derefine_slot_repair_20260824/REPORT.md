# Native-VC derefinement slot repair report

## Verdict

`VC_DEREFINE_SLOT_CORRUPTION_FIXED`

The exact base reproduced the predicted multi-family A5/A6 overwrite.  The
repair stages same-rank vertex-centered derefinement parents in their old
lower-child slots, assembles split-rank parents from deterministic immutable
child sources, and preserves exact injection.  The red test turns green, the
bounded production event is correct through A16 at N128/N256/N512, the early
common-tree convergence gate passes to `t=2.5 M`, and required host/CUDA/MPI
regressions pass.

The final reviewed source is
`6dd20656a305f2543bbbd7001550c6ac67019180`, tree
`551b16fab36ec1d4ce913b39a6478384723aa382`. The numerical production repair
was complete at `d2596707e808aea7ec6167df937d71dc4dbe429e`; later source changes
are default-off diagnostics and focused test/input coverage.

## Defect and repair

At event 3, family 2 had old children 29--32 but new parent GID 26.  Old GID 26
was still a live source for new GID 23.  Writing the injected parent directly
to its new destination during A5 destroyed that live source before A6 copied
it.  This is a deterministic source-slot alias, not a ghost-zone,
coarse/fine-interface, axis-parity, or history-normalization effect.

The same-rank repair makes the old lower-child slot the only authoritative A5
staging location.  For split-rank families, receive metadata identifies each
logical child; local children are snapshotted before A7 can reuse coarse
slots, and A8 assembles each parent with one deterministic writer while
checking coincident shared vertices for consistency.  No averaging, floor,
atomic conflict resolution, or change to the VC injection operator was added.

## Evidence chain

1. Static map reconstruction predicts exactly the old26/new26 collision.
2. The base red test fails with that A5/A6 signature.
3. The minimal same-rank staging repair passes for all 25 variables.
4. The expanded 2D/3D order/transfer matrix passes; a separate 3D selector bug
   was exposed and corrected by testing all active axes.
5. MPI2/MPI4 pure and mixed transactions pass, including an A7 source-slot
   overlap that requires immutable local-child snapshots.
6. One-A100 final-source event-3 writer replays at N128/N256/N512 reproduce
   the same hierarchy checksum and bitwise parent oracles with no modified
   live source; A5--A14 remain exact, A15 changes only projected fields, A16
   preserves every parent and all block-strict interiors, and R0/U0 are finite.
7. The prior catastrophic event-3 constraint injection is absent; the new
   constraint histories are bitwise identical to the earlier repaired runs.
8. Exact 24-event common-tree replay reaches `t=2.5 M` at all resolutions with
   positive finite trusted-core convergence and healthy state/axis diagnostics.
9. The final-source complete enabled host suite passes 137/137; all 20
   targeted final-source Perlmutter CPU/CUDA/MPI checks pass.

## Production impact

Production behavior changes only for native vertex-centered Z4c AMR
derefinement.  Same-rank relocation and split-rank parent assembly are fixed.
The lifecycle and slot-oracle diagnostics are default-off and activated only
by environment variables.  Cell-centered paths retain their original
dispatch and pass the available controls.

## Commit ledger

Each Git commit is the authority for its exact file list.  The table records
the purpose, associated gate, and whether default production behavior changes.

| Commit | Scope and principal files | Test/evidence | Default production change |
|---|---|---|---|
| `6d11a13791369906b94a9503bdd0a4085ceb6adb` | Static map proof under this investigation | Independent event-map reconstruction | No |
| `abe0985ef28430d127b97bc6a5f713f04236ad13` | Multi-family red test and input | Base fails at predicted A5/A6 alias | No |
| `8be0399b0188c6970507d67398df85ea2220b106` | Same-rank VC staging in `mesh_refinement_vc.cpp` | Red test becomes bitwise green | Yes, native VC derefine only |
| `76f05c8598f777b46eebfa4aff9aed42295be2cb` | 2D/3D order/transfer test matrix | Eight same-rank cases | No |
| `941a386f090ff6f04910198ea24bb3f9ee2f12af` | 3D family selector | 3D O2/O4/O6 tests | Yes, 3D native VC derefine only |
| `85ac280d002e200d10d0b684eb661c7421fdf55b` | Split-rank red harness | MPI2/MPI4 expose rank-dependent parent | No |
| `661d25d67451f7a920709efe8b065d0a08028f6c` | Deterministic split-parent assembly | Initial MPI2/MPI4 green | Yes, split-rank native VC derefine |
| `9247a595feb3cd8d2485a23d0e72d5ac3e9c02f5` | MPI output comparison harness | Rank outputs stacked before oracle check | No |
| `a9ff43eb4e88648ef750920543b5ae2d71f3d2d7` | Mixed refine/derefine coverage | Mixed MPI2/MPI4 | No |
| `8ce7b3002d0cbd8760649cf284b7a96646c1c34b` | Targeted A7 overlap fixture | Reproduces local-child reuse | No |
| `8548bc2ffe5a7eef558125c9a472e5ec61504e53` | Immutable local split-child snapshots | Pure/mixed MPI2/MPI4 bitwise green | Yes, split-rank native VC derefine |
| `16afcc45d4e5054e1e29eeb2c2e4438e3e896e24` | MPI red/green report | Records exact rank matrix | No |
| `7927d02fd3fd34fe504dde18998c8a62fde3909e` | A-phase active/interior hash census | Bounded lifecycle fixture | No when diagnostic is off |
| `b2dbde9cf0ca8e4fa6d8eee9bdeff94e130eabeb` | Event-3 replay runner | Authority-bound short replay | No |
| `89622e81495e327be72fc2ff5837650af165e6e0` | VC-native replay diagnostics | Removes incompatible CC-only diagnostic | No |
| `b5e7294ff10909c8cc99c00ccf6a145a185d090e` | Replay authority acknowledgment | Fail-closed source compatibility test | Replay-only contract |
| `597e420a7318ee10b0518ae79a792401f5f03308` | Environment form of source assertion | Supported replay launch path | Replay-only contract |
| `d2596707e808aea7ec6167df937d71dc4dbe429e` | Device pack for lifecycle hashing | Host and CUDA diagnostic fixture | No when diagnostic is off |
| `9dbec92091edb898ccb87fc32bd3044878fbe319` | Replay runner source binding | Clean detached CUDA replay | No |
| `80bc1d74b2fb376d56225cbc18ea5db241e049d2` | Early convergence runner | N128/N256/N512 to 2.5 M | No |
| `479a123cd0b295a0837377fd5198ebeb13fa2719` | Compact evidence, plots, and regression runner | Hash-verified bundles | No |
| `4a7bd38a40cfaf019c4eb66e0aec8d1e0ce75c94` | Qualification documents and manifest | JSON and Markdown validation | No |
| `65af69734798b91d6e2fdf0494393fec67f0fc45` | Move-right and dual split-family ownership fixtures | Layouts C/E on MPI2/MPI4 | No |
| `db6787981b7fd691b55091a6e361879cdd49bec6` | Per-parent A4--U0 writer diagnostic | Final-source event-3 provenance | No when diagnostic is off |
| `6dd20656a305f2543bbbd7001550c6ac67019180` | Fourth deterministic AMR target declaration | Dual-split fixture activation | No |
| `380b40d2975866524c14f19b6ede5c529c85ad94` | Final-source writer, host, CUDA, and MPI evidence | Completion evidence bundle | No |

## Qualification boundary

This result establishes the named slot corruption and the documented early
gate.  It does **not** establish `VC_AMR_QUALIFIED` and does **not** reproduce
Figure 3.  No full collapse endpoint, horizon claim, late-time convergence, or
SYCL runtime qualification was attempted.  The bounded event-3 history samples
are finite-width time brackets, especially at N128.  Exact hierarchy replay is
used only as a control; it is not itself convergence evidence.

The early `t=2.5 M` runs were performed at production-repair commit `d2596707`.
They were not repeated after adding default-off diagnostics and test-only
fixtures; bounded final-source event-3 histories are bitwise identical. Large
restart and binary payloads are retained under the hashed Perlmutter paths and
excluded from Git, while their hashes remain in the manifest.

## Reproduction map

- `STATIC_AUDIT.md` and `EVENT3_MAP.json`: source/data authority and map proof.
- `RED_GREEN.md` and `SAME_RANK_QUALIFICATION.md`: same-rank red/green matrix.
- `MPI_SPLIT_FAMILY.md`: split-rank and mixed-transaction proof.
- `EVENT3_WRITER_REPLAY.md`: A0--A19 writer classification and event ratios.
- `EARLY_CONVERGENCE.md`: three-resolution early gate.
- `PORTABILITY.md`: host, MPI, CUDA, and unavailable-SYCL record.
- `EVIDENCE_MANIFEST.json`: exact hashes, paths, job states, and limitations.
