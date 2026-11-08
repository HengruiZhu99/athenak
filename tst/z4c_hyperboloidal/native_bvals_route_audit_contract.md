# Native boundary-routing metadata audit

This main-only helper links unchanged native runtime objects. It builds the N12
mesh and the actual MappedHalo plan, without allocating physics state, evaluating
an RHS, transporting field values, or evolving. Rebuild does perform its normal
metadata exchanges and allocate plan buffers. A further integer Alltoall checks
the expected send counts against both public MappedExchange statistics.

Eligibility is deliberately conservative and exact: one coalesced donor whose
weight equals one; matching chart, layer, level and node offsets; adjacent native
logical locations; the actual same-level native neighbor GID/level/rank/destination;
target and donor inside the actual generic CC receive/send ranges after translating
origin7 to native origin4; and exact integer index translation. No weight is
snapped. First failed reasons are mutually exclusive. Single-donor nonunit weights
within 1e-12 of one are reported only as a separate diagnostic.

Each rank writes a small block table and eligible-route table, statistics JSON,
and deduplicated donor-use records. The binary header is 16-byte
`BVALS_USES_V1` padded with NULs, uint64 record count, uint32 words=6 and version=1.
Each record is six little-endian int32 values: target GID, donor GID, absolute
donor k,j,i, and usage bits (1 eligible rows, 2 remaining rows). No field values
or full row-weight lists are emitted. Tables cover every native GID exactly once.

Offline analysis regroups records by receiving rank under the independently
verified 24/96 unit-cost GID owners. Unique donors are counted once per receiving
rank, separately from rows and per-block references. A donor used by both row
classes remains in the mapped plan; exclusive eligible donors alone are removable
from that sparse payload. Existing mapped traffic uses 26 residual plus31 physical
Reals, with one payload message per directed peer per exchange per stage. Send
and receive byte totals describe the same network payload and are not added.

Complete native slabs are reported separately, including unused and noneligible
targets and evolved overlap. Generic CC packs the full array field extent, so
using both current31-slot arrays would mean62 fields. A compact26-field residual
staging array would be a later implementation, and57-field slab bytes are only a
labeled hypothetical. This audit neither installs a new route nor establishes
replacement equivalence or performance improvement.

CPU captures describe that compiler's host-generated plan. Projected24/96 counts
use known ownership but are not actual Aurora measurements; strict coalescing can
depend on host arithmetic. An Aurora capture must use its own frozen CMake recipe.
The portable builder replaces only main.cpp.o, preserves numerical source/object
and executable hashes, and never submits jobs. Runtime: native_bvals_route_audit
INPUT in a new output directory. Process exit0 and the final PASS line are required
in addition to all rank records.
