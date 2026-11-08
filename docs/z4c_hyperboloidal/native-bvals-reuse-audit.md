# Native boundary-routing reuse audit

The metadata-only CPU audit admits 76,032 of 481,248 bulk halo rows (15.799%)
as exact one-donor/unit-weight, same-chart/layer/level native-neighbor routes
within ordinary CC send/receive index boxes. All admitted rows are in the
Cartesian chart. The mutually exclusive first failures are 362,180 multi-donor
rows, 19,420 different-layer single-donor rows, and 23,616 single-donor rows outside
the nominal native receive box. No weight was rounded into eligibility.

The two-rank capture constructed the actual frozen MappedHalo plan. Independently
counted unique donors, send/receive donors and peers matched both public exchange
statistics. It allocated no physics state, transported no field values and ran
no RHS or evolution. Runtime sources, objects and the original executable remain
unchanged. The helper emits only aggregate tables and 941,090 deduplicated
target-block/donor-use records, rather than 28,349,184 row-weight entries.

The following counts project this CPU host-generated plan onto the 24/96-rank
GID owners already verified from Aurora geometry traces. They are not actual
Aurora donor-plan or performance measurements. Payload is counted once at the
receiver, per stage, excluding headers and setup metadata.

| Quantity | 24 ranks | 96 ranks |
|---|---:|---:|
| Eligible remote rows |51,744|76,032|
| All unique remote donors |635,325|871,908|
| Eligible-exclusive remote donors |42,650|76,032|
| Donors shared with remaining rows |3,790|0|
| Current mapped payload, bytes |289,708,200|397,590,048|
| Removable sparse payload, bytes |19,448,400|34,670,592|
| Removable sparse fraction |6.713%|8.720%|
| Complete native slab points |68,736|96,768|
| Unused points in those slabs |16,992|20,736|
| Native slab payload with 62 fields, bytes |34,093,056|47,996,928|
| Remaining sparse + full 62-field slabs, bytes |304,352,856|410,916,384|
| Increase over current payload |5.055%|3.352%|

The current mapped transport sends 26 independent residual and 31 physical fields.
Generic CC packs the full field extent of each array, so directly using the two
existing 31-slot arrays means 62 fields. Even a hypothetical compact 26-field
residual pack leaves complete-slab payload overhead; the report retains that
57-field variant separately. Shared donors stay in the sparse plan, and repeated
donors are deduplicated per receiving rank, not per row or block.

This gives a quantitative reason to consider selective native routes rather
than assume whole native slabs reduce traffic. It establishes no replacement
equivalence or speed benefit. Surface exchanges, coarse/fine interpolation and
cross-chart transport are outside this eligibility claim. Exact Aurora counts
would require the same helper built with the frozen Aurora CMake recipe because
host arithmetic can affect coalesced donor weights.

Evidence: [CPU audit](evidence/native-bvals-route-cpu-audit.json), with source,
build, execution and data hashes. Detailed per-rank and directed-peer results
are retained at `build-native/native-bvals-route-projection-v1/results.json`.
The independent Tasks-agent source review found no blocking accounting or index
translation defect. No new transport has been implemented.

One separate hypothesis is visible in the current `FormResidual` source: each
independent donor residual is `(state-reference)/scale`, where scale is the
donor's `r*r` for chi and one otherwise. If profiling identifies mapped traffic
as material, examine sending the31 raw state fields once and reconstructing
the26 residual values per donor from an exact cached donor reference and scale.
That could reduce this payload from57 to31 scalars per donor before accounting
for cache setup and reconstruction. The operation must occur before interpolation;
subtracting a reference after interpolation would change the chi calculation.
Raw fields must remain available unchanged for KO. Exact reference/scale bytes,
operation order and cache invalidation after regridding would need validation.
This is a source-derived hypothesis, not an implemented or measured improvement.
