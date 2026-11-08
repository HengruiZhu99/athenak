# Actual Aurora stage and decomposition checks

The frozen native numerical objects pass the first-RHS, one-step and eight-step
comparison against the unchanged predecessor on both 24 and 96 MPI ranks. Every
comparison covers 229392 evolved bulk points, including overlap, and 864 owned
scri points. Runtime sources and the original executables remain unchanged.

The test-only predecessor accessor was first admitted against the unchanged
original executable: its initial and final-eight-step snapshots are byte-identical
in all 31 fields. The native first RHS is also byte-identical to the actual old
GPU first RHS: 7,111,152 bulk and 26,784 scri scalar rates per decomposition.
The accessor uses the pinned source and archived FP/device flags; it is not a
new CPU reference evolution.

| Comparison to original GPU | Bulk maximum difference | Scri maximum difference |
|---|---:|---:|
| Initial state |0|0|
| First RHS |0|0|
| Accepted step 1, including direct unchanged one-step control |2.168404344971009e-19|0|
| Accepted step 8 |5.551115123125783e-17|4.342378097006337e-17|

These values are the same at 24 and 96 ranks. All use the predeclared short field
and separate chi gates. All eight actual old per-cell ChiRecord members at
accepted0/1/8 agree byte-for-byte with native: 1,835,136 bulk and 6,912 scri record
scalars per snapshot. Actual bulk and scri coordinates agree byte-for-byte.
Every step's maximum clock residual agrees exactly with the immutable original
per-step CSV. Accepted states and post-diagnostic snapshots remain explicitly
distinguished in the reports.

For native clean versus poisoned execution at each decomposition, and for
native 24 versus 96, all 26 region/label datasets and all 32 stage records are
byte-identical. This includes the first four stage inputs/outputs with all 31+8
values, accepted0/1/8, first RHS and actual geometry. Poisoning covers unowned
bulk u0 ghosts and the public residual/RHS/status workspaces while preserving
evolved overlap; it does not claim to poison private exchange/surface scratch.

The first mapper job failed during test-fixture teardown after writing its
metadata; no evolution had run. Its artifacts remain unadmitted. The successful
retry changed only the helper's initialization of an unused coordinate pointer.
The native and predecessor observer executables were unchanged.

The original tracer does not record CFL. A separate read-only observer of the
original native main now passes the complete 24/96-rank qualification. At each
decomposition its initial and eight-step checkpoints preserve all 24,704,784
padded 31+8 scalars exactly, and its ordinary history, horizon summary and full
harmonic shape files are byte-identical to the original native main. All eight
clock and CFL maxima are byte-identical to the unchanged original GPU solver.
The largest compatibility residual is 9.778256860928204e-16.

The observed 24/96-rank states and all 32 stage records agree exactly. A two-step
prefix at 24 ranks followed by a restart at 96 ranks gives the identical final
state and identical joined stage records to continuous execution. The ten final
history values agree exactly; areal radius and irreducible mass differ by
6.13389339321202e-11 and 3.06694669660601e-11, respectively, within the adopted
diagnostic gates. The additional horizon search at the end of the restart prefix
remains an explicit diagnostic phase difference. The observer adds
synchronization and is unsuitable for performance measurements. These short
checks do not establish long-trajectory agreement.

Evidence: [24-rank science](evidence/aurora-stage-trace-two.json),
[96-rank science](evidence/aurora-stage-trace-eight.json), and
[native decomposition parity](evidence/aurora-stage-trace-cross-rank.json), and
[observed-main and restart qualification](evidence/aurora-step-observer-science.json).
Each compact record links and hashes its full report, whose pointwise statistics,
source/input provenance and worst locations remain available.
