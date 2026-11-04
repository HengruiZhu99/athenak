# Native participation and buffer gates (before changing native code)

The existing equal native costs, capacity, rank guard, prolongation and migration
remain unchanged in M0/M1. The review's [10,1,1] reproduction is a metadata bug,
not permission to exercise invalid native ownership in the evolution driver.

A future native change must meet all these gates:

* For zero/one/several skipped ranks (including [10,1,1] on three ranks), initialize
  every count and displacement; check monotone prefix sums, exact total coverage,
  contiguous ownership and rank_eachmb consistency. Check zero, tied and highly
  skewed costs, more ranks than leaves, and invalid/nonfinite costs collectively.
* Build admissible balanced trees with refinement adjacent to boundaries and
  unequal assignment. Enumerate actual routed child/parent messages by destination.
  Sum each packed state plus compensation offset. Verify metadata capacity and
  send/receive extents independently, including eight remote child messages and
  multiple parents. For N12/ng4 a child is 2744 values/field and a fine slot 8000;
  the former cannot be bounded by the latter without counting messages.
* Preflight topology without mutating the live tree, or stage a rollback-capable
  transaction. Reject insufficient capacities collectively before publication;
  retain old arrays, tree, metadata and sequence. Test injected failure on one rank.
* Test ownership-only rebalance explicitly. Compare all31 and nonzero compensated
  fields after refine/coarsen/migration, changed-rank restart, and no-change events.
  Run main Z4c AMR regressions before accepting any generic change.

Design alternatives remain open: explicit empty-native ranks require auditing all
native zero-sized paths; a Cartesian communicator requires mapping all collectives
and migration lifetimes; prepartitioned initialization alone cannot accommodate
later coarsening below the world size. There is no safe mixed-AMR >8-rank claim yet.
