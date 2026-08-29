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

## Multi-rank measurement

The first optimized strong-scaling curve confirms that the established
multi-rank path is unacceptable.  On the same 212-MeshBlock frozen workload,
the scaling-allocation one-tile rate was `4.953e6 zone-cycles/s`; 24 tiles
reached only `4.282e6`, or `0.864x` speedup and `3.60%` efficiency.  Removing
file-output time does not rescue the curve: one tile took `11.760 s` and 24
tiles took `13.968 s`.  Thus the result is not merely parallel filesystem
noise.

The source audit found that the multi-rank path performs, on every one of the
roughly ten reconciliations per RK cycle:

1. a device pack and fence;
2. full packed-table D2H staging;
3. global host `MPI_Allgatherv`;
4. redundant reconstruction of every global group on every rank;
5. local replacement H2D staging and application;
6. an exact device postcondition, D2H result copy, and `MPI_Allreduce`.

Step 6 was also an implementation inconsistency: `lean_runtime` disabled the
observational exact postcondition on one rank, but the multi-rank branch
ignored that option.  Commit `4260e5ba` makes the MPI branch honor the same
default-off lean postcondition while retaining it for exhaustive/default runs
and explicit localization diagnostics.  This removes redundant audit work but
does not alter the canonical averaging or eliminate the global host-staged
path.

## Sparse production repair

Commit `62993e7b` implements the topology-rebuild-time sparse plan:

- select one deterministic owner rank per canonical group;
- pack only finest-level authoritative contributors into persistent device
  buffers in canonical order;
- exchange only with ranks sharing those groups, using GPU-aware MPI;
- have the owner compute the ordered arithmetic mean;
- return one group value only to participant ranks;
- apply from persistent device buffers without host staging.

The topology rebuild may still gather metadata globally, but the RK hot path
does not stage field values through the host and does not use
`MPI_Allgatherv`.  Each canonical group has the rank of its first finest-level
canonical contributor as owner.  Phase A sends only authoritative values to
that owner through persistent device buffers; the owner sums them in the
unchanged canonical order.  Phase B returns one mean only to ranks that store
a contributor, and a device kernel applies it to every local copy.  A private
duplicated communicator isolates the two point-to-point phases from all other
AthenaK traffic.

The old exhaustive path remains available whenever the localization CSV or
exact synchronization postcondition is requested.  This preserves the
debugging contract without putting global host staging in lean production.

## Bounded Aurora discriminator

Aurora job `8790725` tested source `62993e7b` at 2 and 24 ranks, one PVC tile
per rank, over the same frozen 212-MeshBlock N512 window. Job `8790735` added
the required 12-tile one-node point. All cases reached `t=9.85 M` with exit
zero.

| tiles | prior lean throughput | sparse throughput | throughput gain | prior non-output | sparse non-output | non-output gain |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.702151e6 | 2.558434e6 | 1.503x | 18.631 s | 16.838 s | 1.106x |
| 12 | 4.182024e6 | 4.815004e6 | 1.151x | 14.550 s | 6.765 s | 2.151x |
| 24 | 4.391482e6 | 6.974422e6 | 1.588x | 12.549 s | 6.644 s | 1.889x |

The complete histories are byte-identical to their matched pre-sparse runs.
All matched final restart numerical payloads have SHA-256
`e498614cad5e50677a1698bc20680e5b34131e0a098c3421e1d64564702c6ab6`.
Thus the exchange preserves the canonical synchronization result in the
tested production trajectory.

The repair is material but does not make this small hierarchy scale well.
Against the final-source one-tile point (`5.958188e6`), the sparse 24-tile
result is only `1.171x` aggregate speedup, or `4.88%` parallel efficiency.
The non-output speedup is `1.767x`, still only `7.36%` efficiency. At only 8.83
MeshBlocks per rank, fixed per-rank task launches, the remaining boundary
exchanges and reductions, and low GPU occupancy now dominate. The evidence
does not assign those residual costs individually.
