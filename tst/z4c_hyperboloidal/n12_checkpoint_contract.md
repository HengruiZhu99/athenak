# Native N12 checkpoint export and comparison contract

The decoder accepts only fixed-N12 native shared restart files, with all 106
native leaves, 229392 evolved bulk points, 864 owned scri points, 31 stored
fields and eight ChiRecord members per point. Writer MPI rank count is metadata,
not a field-order assumption; the same decoder supports 24/96-rank shared files.
It rejects per-rank files and changed/adaptive trees instead of guessing a map.

The one-time mapping helper uses the original native Mesh/tree/atlas, with no
native physics allocation or evolution. The independent legacy N12 model
supplies active-point and surface-row order, initial fields, evolved-point
volumes and surface areas. Match every leaf's face/layer/exact bounds bijectively;
require byte-identical float64 coordinates at every bulk and scri sample.
Prove complete one-to-one native padded restart slots and canonical ordinals.
The portable mapping sidecar is bound to exact topology, atlas and numerical
policy identities from a known native initial checkpoint, with source/object,
transitive legacy-header, helper executable and template-checkpoint hashes.

Coordinates are not stored in native restart payloads. Matching checkpoint
metadata to this sidecar proves the host atlas construction, not new GPU map
arithmetic. Optionally supply all `NATIVE_DIAG_V1` rank dumps: validate their
legacy ordinals/native keys and actual cached bulk coordinate bytes. Those
geometry caches are computed on host before device upload. Scri coordinate
proof remains the exact sidecar construction.

Require unchanged N12 model, fixed tree, physics identity, KO=.1, chi floor
1e-10, requested dt=.00046944203735617445, classical RK4, float64, constant
shift policy and fixed-origin horizon settings if present. Check exact cycle
and time from repeated additions of the nominal dt; do not accept a clipped
5M endpoint. Bound all byte lengths before reading, require exact payload size,
finite values including padding, zero unused slots, positive current chi and
nonnegative integral chi-event counters. Export all31 fields and all8 records
in the pinned legacy `fields.bin`/`boundary.bin` order.

Compare every stored field, including dependent/dormant fields, at every bulk
and scri point. Through eight steps use the adopted `1e-12+5e-10*abs(reference)`
gate, with chi `1e-14+5e-10*abs(reference)`. At step3551 use
`1e-10+1e-8*abs(reference)`, with chi `1e-13+1e-8*abs(reference)`. Require the
pinned complete step3551 reference hashes. Report full coverage, finite checks,
bitwise equality, per-field and total absolute/scaled/tolerance-ratio maxima,
worst values/coordinates/indices, and volume/area-weighted RMS. Full-state error
weights include evolved overlap, matching the existing trajectory oracle;
zero diagnostic quadrature weight never suppresses an evolved-field comparison.

Step3551 has only aggregate `chi-policy.json`, not per-cell reference histories.
Export all native ChiRecords, compare the available aggregate minima/events/
corrections/first-floor/radius quantities and report differences. Never invent
per-cell reference histories or claim their comparison. Counts near thresholds
are reported separately, as required by the adopted protocol. Stage normal
compatibility, clocks/CFL and common-time constraint/horizon histories are not
stored in restart payloads and require separate evidence.

Tool validation uses actual native initial checkpoints plus the already
independently validated CPU eight-step canonical trajectory fixtures. Include
late-field/scri/chi threshold and malformed-metadata/truncation negative cases;
do not infer correctness only from equal global norms. No remote job or full
local3551 evolution is part of this tooling task.

The first writer-fixture validation identified an existing native header detail:
`Mesh` initializes the global fine indices and all block indices, but leaves the
nine unused global coarse-index members unspecified. `RestartOutput` serializes
the whole struct. The decoder therefore compares every initialized header field
exactly and reports those nine unused integers without comparing their values;
it does not alter runtime initialization or ignore any used mesh dimension,
topology, geometry, timestep or numerical-policy field. The original v1 test
failure is retained. This exception is required for portable native headers,
not a relaxation of the scientific field/coordinate gates.

Cross-platform diagnostics are a separate script,
`n12_checkpoint_cross_platform.py`. It reports exact numerical-identity and
coordinate mismatches as failures, validates every other header condition,
then measures all fields with unchanged short gates. It cannot turn failed
strict admission into a pass. Through eight steps, compare complete available
CPU ChiRecords: the first two raw minima use chi's short gate, while the six
zero-event/count/correction/first-time entries must agree exactly. The actual
Aurora initial/eight audit retains both exact-admission failures and the
eight-step field failures; same-platform original GPU control is pending.

The subsequent unchanged original GPU24 control passes all nativeGPU24
initial/eight field gates, exact bulk coordinates and common serialized
resolved-policy doubles. Preserve the failed CPU comparison alongside this
matched-backend result. `n12_checkpoint_gpu_control.py` accepts24/96 ranks,
requires pinned original input/configuration/executable provenance and SP1
state consistency, and compares only available legacy chi aggregates.
`n12_checkpoint_rank_compare.py` instead compares two complete native ledgers
and all fields bitwise, with exact same-backend policy identity. Neither tool
relaxes the existing runtime restart identity guard. Helper negative controls
cover a late scri field, nonzero chi events/corrections, and malformed actual
coordinate membership.
