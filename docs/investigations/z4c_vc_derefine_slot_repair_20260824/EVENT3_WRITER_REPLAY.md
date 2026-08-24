# Event-3 writer replay

## Scope and authority

This is a bounded, one-rank replay of the authenticated common-tree event at
coordinate time `0.2979919496568637 M`. The final writer replay uses source
commit `6dd20656a305f2543bbbd7001550c6ac67019180`, exact VC injection, O4 bulk
operators, q6 transfer, RK4, CFL 0.15, KO 0.02, the existing telegraph lapse and
Gamma-driver shift, and zero Z4 damping.  No numerical parameter or hierarchy
event was changed.

The accepted event-3 tree checksum is `b6c60694c30a2049`.  The replay was run
at N128, N256, and N512 in Perlmutter one-A100 `shared_interactive`
allocations. All three final-source executions ran sequentially in job
`57505077`, used one A100 and at most 16 active CPU cores, and returned zero.

## Slot oracle

Event 3 collapses two four-child families:

| Family | Old child GIDs | Canonical staging slot | New parent GID |
|---|---:|---:|---:|
| 1 | 16--19 | old slot 16 | 16 |
| 2 | 29--32 | old slot 29 | 26 |

Old GID 26 is live before the transaction and maps to new GID 23.  The broken
path wrote family 2's parent into destination slot 26 during A5, before A6 had
copied that live old block.  The repair stages each same-rank VC parent in its
old lower-child slot and lets A6 perform the logical relocation.

At all three resolutions and for all 25 evolved variables:

- A5 staging equals an independently injected parent bitwise;
- A6's relocated parent equals that oracle bitwise;
- no live old GID is modified by A5;
- no unaffected old block is corrupt after A6.

The N256 family-2 oracle and A5 staging hash are both
`ff0a63556d545282`; the final A6 parent has the same hash.  The destination's
A5 hash differs, as it should, because new GID 26 is not authoritative until
A6.

## Per-parent writer and lifecycle census

The final-source replay records two parents times nine named checkpoints, or
18 writer records per resolution. For both parent locations `(4,2,7,0)` and
`(4,2,8,0)`, at N128/N256/N512 and for all 25 variables:

- A5 staging equals the independent injection oracle bitwise;
- A6, A8, and A14 parent hashes remain equal to that oracle;
- all relocation survivors are exact;
- the first and maximum numerical mismatch records are absent;
- A15 changes exactly `gxx`, `gxy`, `gyy`, `gzz`, `Axx`, `Axy`, `Ayy`, and
  `Azz`, as required by algebraic projection;
- A16 does not change either derefined parent's active hash relative to A15;
- R0 and U0 are present at RK stage 1 and contain no nonfinite values.

Across the complete MeshBlock pack, A15--A16 full-active hashes change for the
same eight projected variables at shared block-boundary vertices, while every
block-strict-interior hash remains unchanged. This is the expected boundary
cache/ghost reconstruction distinction; it is not an independent-interior
writer.

The fail-closed analyzer is `analyze_event3_writer.py`; its compact output is
`evidence/analysis/event3_writer/summary.json` and reports `PASS` for all
three resolutions.

## Earlier full lifecycle census

At event 3, the N128/N256/N512 lifecycle classifications agree:

- A5 writes only the injected derefinement parents into their old lower-child
  staging slots.
- A6 performs the intended relocation.
- A7 and A8 do not change the one-rank active state.
- A14 to A15 changes only algebraically projected variables 1, 2, 4, 6, 8, 9,
  11, and 13.
- A156 to A157 changes those variables' full-block hashes while their strict
  interior hashes remain unchanged; this is boundary prolongation, not an
  interior writer.
- A159 to A16, A16 to A17, and A17 to A18 do not change block-strict interiors.

The complete A0--A19 and A150--A159 records are retained as JSONL rather than
summarized away.

## Constraint bracket

The history samples bracket the event rather than sampling the state at an
infinitesimal pre/post transaction time.  N128's broad bracket also crosses the
next closely spaced hierarchy event, so its ratios are conservative and are
not attributed solely to event 3.

| Resolution | C post/pre | H post/pre | M post/pre | Z post/pre |
|---|---:|---:|---:|---:|
| N128 | 0.94275 | 1.03915 | 0.72697 | 0.97962 |
| N256 | 0.62144 | 0.51307 | 0.45335 | 0.97389 |
| N512 | 0.82283 | 0.78531 | 0.90248 | 0.94284 |

There is no many-orders-of-magnitude injection.  N512 gives the narrowest
bracket (`0.2979278648` to `0.2988891371 M`) and all four norms decrease.

## Evidence boundary

Observation: the predicted A5/A6 corruption is absent after the repair, both
parents remain correct through A16, and the first subsequent RHS and RK update
are finite in all three bounded runs. The new constraint files are bitwise
identical to the earlier repaired replay despite enabling the additional
default-off writer audit.

Inference: the slot overwrite was the cause of the previously observed
event-3 resolution-growing pulse.

This replay does not qualify arbitrary VC AMR histories, long-time Brill
collapse, or Figure 3.

Primary final-source evidence is under
`evidence/perlmutter/event3_writer_6dd_*`; the compact plot is
`evidence/analysis/early_history/figures/event3_constraint_jump_ratios.png`.
