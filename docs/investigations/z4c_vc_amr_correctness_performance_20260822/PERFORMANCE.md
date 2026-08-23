# Performance disposition

## Status

`VC_Z4C_PERFORMANCE_QUALIFIED = NO`

No profiling, CC/VC throughput comparison, or performance rewrite was
performed.  The controlling goal forbids performance work until every
correctness, restart, MPI, physical-regression, and backend gate is green.
Current-source CUDA and physical Brill/common-tree gates remain open.

Consequently there are no defensible values for:

- effective zone-cycles/s;
- active-point updates/s;
- CC/VC throughput ratios;
- per-step kernel/communication/transfer time;
- memory footprint comparisons;
- before/after profiles.

The current shared-node implementation still uses host mirrors and
`MPI_Allgatherv`.  These are recognized performance targets, not correctness
defects.  Replacing them before qualification would mix numerical and
performance semantics and was deliberately avoided.

`profiles/README.md` records why the profiles directory is intentionally empty.
