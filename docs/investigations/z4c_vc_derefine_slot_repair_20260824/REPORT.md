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
6. One-A100 event-3 replays at N128/N256/N512 reproduce the same hierarchy
   checksum and bitwise parent oracles with no modified live source.
7. The prior catastrophic event-3 constraint injection is absent.
8. Exact 24-event common-tree replay reaches `t=2.5 M` at all resolutions with
   positive finite trusted-core convergence and healthy state/axis diagnostics.
9. The complete enabled host suite and selected CUDA/MPI production-path
   regressions pass.

## Production impact

Production behavior changes only for native vertex-centered Z4c AMR
derefinement.  Same-rank relocation and split-rank parent assembly are fixed.
The lifecycle and slot-oracle diagnostics are default-off and activated only
by environment variables.  Cell-centered paths retain their original
dispatch and pass the available controls.

## Qualification boundary

This result establishes the named slot corruption and the documented early
gate.  It does **not** establish `VC_AMR_QUALIFIED` and does **not** reproduce
Figure 3.  No full collapse endpoint, horizon claim, late-time convergence, or
SYCL runtime qualification was attempted.  The bounded event-3 history samples
are finite-width time brackets, especially at N128.  Exact hierarchy replay is
used only as a control; it is not itself convergence evidence.

## Reproduction map

- `STATIC_AUDIT.md` and `EVENT3_MAP.json`: source/data authority and map proof.
- `RED_GREEN.md` and `SAME_RANK_QUALIFICATION.md`: same-rank red/green matrix.
- `MPI_SPLIT_FAMILY.md`: split-rank and mixed-transaction proof.
- `EVENT3_WRITER_REPLAY.md`: A0--A19 writer classification and event ratios.
- `EARLY_CONVERGENCE.md`: three-resolution early gate.
- `PORTABILITY.md`: host, MPI, CUDA, and unavailable-SYCL record.
- `EVIDENCE_MANIFEST.json`: exact hashes, paths, job states, and limitations.
