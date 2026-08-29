# Native-VC shared-vertex synchronization audit

## Required semantics

All stored copies of a coincident physical vertex must be replaced by one
deterministic value. The existing authority rule is:

1. identify all contributors with the same canonical physical-vertex key;
2. retain only contributors at the finest level present in that group;
3. visit those contributors in the established canonical order;
4. compute one arithmetic mean per evolved variable;
5. write that value to every local contributor, including coarser copies.

This reconciliation is part of the native vertex-centered numerical
representation and is not an optional diagnostic.

## Unoptimized implementation

Every synchronization call packed all local contributors on device, fenced,
copied the complete packed table to the host, built host vectors, used global
host-staged `MPI_Allgatherv`, computed all group averages on the CPU, copied
the local replacements back to device, applied them, and ran a second
device/host postcondition. Even on one rank the full staging path remained.

For the profiled N512 interval, this occurred 212 times in 21 cycles. Each
call included an approximately 9.68 MB D2H contributor copy. This was a direct
source of GPU serialization, allocation traffic, and host work.

## One-rank optimized path

At topology-plan rebuild, the candidate constructs persistent device arrays
containing:

- each local contributor's canonical group;
- a flattened list of authoritative contributor indices in the exact existing
  canonical order;
- begin/end offsets for each group;
- persistent group-value storage sized for all Z4c variables.

At runtime, one device kernel computes the ordered finest-level averages and
a second applies them. Kernel ordering supplies the dependency; no host fence,
D2H copy, host reduction, H2D copy, or temporary allocation is required. A
default-off postcondition can still verify every replacement.

The implementation does not assume that a group's contributors are contiguous
in MeshBlock-local storage; the flattened canonical index list makes that
mapping explicit.

## Multi-rank status

The candidate deliberately leaves the established multi-rank path unchanged
until the optimized strong-scaling curve is measured. That path still uses a
global host-staged `MPI_Allgatherv`. It is correct but is not an acceptable
final production scaling design if measurements confirm that it dominates.
The intended replacement is a precomputed sparse neighbor exchange with
persistent device buffers and deterministic canonical reductions, preserving
the authority rule above.
