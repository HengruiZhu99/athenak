# Native-VC AMR stage order and T0-T9 localization

## Evolution-stage contract

At each RK stage the native-VC path follows this logical order:

1. expose/copy the accepted or accumulated stage state;
2. reconstruct Cartoon-axis parity data when applicable;
3. evaluate the Z4c RHS;
4. apply RHS physical boundary conditions;
5. perform the RK state update on active vertices;
6. reconcile shared active vertices;
7. inject fine coincident vertices into the coarse cache;
8. exchange same-level and coarse-cache boundary data;
9. fill physical/axis coarse ghosts;
10. prolong dependent hanging and fine ghost vertices;
11. restore axis regularity/parity before the next RHS.

At an accepted regrid/load-balance event, topology keys, contributor groups,
boundary buffers, and device-visible metadata are rebuilt.  Accepted-state
algebraic projection is applied to canonical values and dependent/cache data
are reconstructed afterward.

## Diagnostic checkpoints

| checkpoint | observed state |
|---|---|
| T0 | state immediately after regrid population |
| T1 | shared-node reconciliation |
| T2 | fine-to-coarse injection |
| T3 | same-level/coarse-cache communication |
| T4 | physical-boundary/coarse-ghost fill |
| T5 | coarse-to-fine hanging/ghost prolongation |
| T6 | Cartoon-axis parity/regularity reconstruction |
| T7 | RHS evaluation |
| T8 | block-local RK update before reconciliation |
| T9 | post-stage canonical state |

## First failing operation before repair

Canonical shared active copies were exact after T0/T1.  The first
resolution-worsening discrepancy was created at T5 in the lower-side first
ghost layer.  It was not a topology-placement or initial same-level-copy
failure.  T7 converted that wrong ghost value into an approximately `O(1/h)`
`Atilde_zz` RHS mismatch, and T8 then changed active state.

The static code defect was `offset / 2` for negative odd offsets.  Correct
floor division moves `-1` from coarse interval `[0,1]` to `[-1,0]`.  After
that fix, the remaining about-second-order discriminator localized to
insufficient transfer accuracy rather than stage freshness.  Raising transfer
orders to q4/q6/q8 restores measured bulk-order behavior.

Default-off diagnostic environment variables preserve this localization
without changing ordinary runtime behavior.  Raw contributor/state/RHS files
under `evidence/phase1/` and `evidence/phase2/` are compressed because the
largest records are tens of megabytes.
