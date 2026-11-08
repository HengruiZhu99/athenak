# Fixed-N12 native checkpoint comparison

`tst/z4c_hyperboloidal/n12_checkpoint.py` reads an ordinary native **shared**
restart, exports all 31 bulk/scri slots and eight ChiRecord members in pinned
legacy sample order, and optionally compares the full fields to a reference.
Its indexing uses native leaf GID and padded record slots, independently of
the writer's MPI decomposition. Per-rank files and changed/adaptive trees are
rejected; they need a separately proven mapping.

The reusable local mapping is `build-native/n12-checkpoint-map-v1`. Preserve
`mapping.npz` and `manifest.json` together. The generator matched all 106 native
leaves to the independent legacy N12 model and checked every coordinate byte
for 229392 bulk and 864 scri points. Its source, transitive legacy dependencies,
native objects, executable and initial-checkpoint hashes are recorded in the
manifest. The proof includes exact topology/atlas/policy identity and complete
one-to-one mapping coverage.

For the eventual native step3551 restart:

```sh
python3 tst/z4c_hyperboloidal/n12_checkpoint.py \
  --candidate /path/to/native-step3551.rst \
  --mapping build-native/n12-checkpoint-map-aurora-v1 \
  --reference build-native/n12-reference-step3551 \
  --step 3551 --output build-native/native3551-comparison
```

The output directory must be new. This reads the source checkpoint without
altering it. Omit `--reference` to export without claiming a field comparison.
The tool enforces full nominal steps, exact dt and coordinate time, the fixed
tree, all meaningful mesh dimensions, the numerical-policy identity and pinned
N12 input values. Step3551 reference fields, boundary, time and chi-policy
sidecars must retain their pinned hashes. A clipped endpoint at exactly 5M is
rejected.

The exported files are point-major float64:

| File | Shape and contents |
| --- | --- |
| `fields.bin` | 229392 ×31, legacy active bulk-point order |
| `boundary.bin` | 864 ×31, legacy scri-row order |
| `chi-records.bin` | 229392 ×8, matching bulk order |
| `surface-chi-records.bin` | 864 ×8, matching scri order |

The eight record columns are candidate minimum, accepted-candidate minimum,
floor count, nonpositive count, nonfinite count, maximum correction, summed
correction and first-floor coordinate time. Native `1e300` minima/first-floor
values are untouched storage sentinels. The summary reports no first-floor
time when the event count is zero. All31 slots already use the same stored
Cartesian-component conventions in both implementations; no face-dependent
tensor rotation or change to lapse/shift/trace offsets is introduced.

Every field is compared, including dependent and dormant slots and evolved
overlap cells. Short runs use the adopted arithmetic/chi gates; step3551 uses
the adopted long/chi gates. `results.json` contains complete counts, failed
values, bitwise equality, per-field and total absolute/scaled errors, tolerance
ratios, worst coordinates/indices/values and weighted RMS. Error weights are
the legacy evolved-point volumes, including overlap and the nodal half weight,
and scri areas, matching the existing trajectory oracle. They are separate
from the unique-coverage weights used for physical constraint histories.

The step3551 chi-policy reference is **aggregate only**. The tool compares and
reports available event/minimum/correction/first-floor aggregates, while
explicitly marking per-cell reference histories unavailable. It exports all
native records without inventing reference values. Stage normal compatibility,
clocks/CFL and common-time constraint/horizon histories require separate
evidence; they cannot be reconstructed from this checkpoint alone.

Native restart payloads contain no coordinates. The sidecar binds the exact
host atlas construction to the checkpoint metadata. To audit actual cached
bulk coordinates from a native replay, add all rank dumps:

```sh
  --coordinate-dumps /path/to/points-rank*.bin
```

The `NATIVE_DIAG_V1` importer requires each legacy ordinal exactly once and
checks native GID/indices and all cached coordinate bytes. These caches are
constructed on host before GPU upload; this is not a claim of GPU map
generation. The exact sidecar proof covers scri coordinates.

Local validation passed for ordinary native initial checkpoints written on two
and eight ranks against the independent legacy initial fields. A separate
format fixture imports the already validated eight-step CPU trajectory into
native storage and writes through the actual `RestartOutput`; it performs no
new evolution. Complete fields and records survive export bitwise at both
decompositions. Targeted negative tests cover the last scri field, evolved
overlap with zero diagnostic weight, chi's tighter gate, nonfinite data,
truncation and wrong timestep/topology/policy. Synthetic 24/96-rank **metadata**
controls verify decomposition-independent shared-file decoding; they are not
substitutes for actual GPU runs.

The first format test also exposed an existing native header detail: nine
unused global coarse-index integers are uninitialized but serialized by the
original writer. The tool reports those values and compares every initialized
header field exactly. The v1 failure is retained; the v2 correction changes
neither runtime code nor any scientific comparison tolerance. Evidence resides
in `build-native/n12-checkpoint-validation-v2/results.json` and the two
`build-native/n12-checkpoint-initial-*-v1/results.json` files.

## First actual Aurora 24-rank checkpoint audit

The initial/eight-step shared checkpoints from job9051398 decode completely:
229392 bulk points, 864 scri points, all31 fields and all8 ChiRecord members.
Exact time, nominal dt, tree, input parameters, finite values and unused
payload padding checks pass. This was an actual GPU executable run, separate
from the local writer-format fixtures above.

The exact CPU sidecar numerical-identity check fails: nine derived gauge/profile
coefficients differ by 1–3 ULP. The strict decoder failure remains recorded in
`build-native/aurora-native-{initial,eight}-24-v1/failure-audit.json`.
No runtime restart identity guard or comparison tolerance was changed.
`n12_checkpoint_cross_platform.py` is a separate diagnostic that explicitly
retains this failed admission check while evaluating the complete fields.

| State | Bulk failed values | Scri failed values | Bulk max absolute error | Scri max absolute error |
| --- | ---: | ---: | ---: | ---: |
| Initial | 0 | 0 | 1.1457501614e-13 | 2.2204460493e-16 |
| Eight full steps | 381192 | 13994 | 4.8918368549e-9 | 1.3188657371e-9 |

The eight-step comparison **fails the adopted short field gates** against the
independent CPU Step3D reference. Its bulk/scri weighted RMS errors are
9.7240840401e-11 and 1.5440564813e-10. Largest errors occur in the A tensor,
connection/D fields and Theta near the outer region. This failure is not
explained away by a successful smoke run or finite values. A same-platform
comparison with the unchanged original Aurora GPU executable is required
before attributing the discrepancy to cross-platform arithmetic or migration.

All1,842,048 chi-record scalars pass the short-run record comparison: the two
raw minima use the tighter chi gate; event counts, correction sums/maxima and
first-event sentinel match the zero-floor reference exactly. There are no
floor, nonpositive or nonfinite events. At cycle8,
t=.0037555362988493960 and minimum chi=1.342207406105154e-5.
The per-cell reference here is the CPU native trajectory already checked
bytewise against independent legacy records. This does not create per-cell
reference histories for the historical step3551 run.

Actual cached coordinates from the frozen native Aurora saved3551 replay
cover every bulk key. Relative to the CPU sidecar,72030 coordinate components
across54072 points differ, with maximum absolute difference2.2204460493e-16.
The exact-coordinate gate therefore also fails across platforms. Aurora
device/host r>.94 masks differ from the CPU-map mask at118 positive-weight
points (54 Aurora-only,64 CPU-only). The coordinates were generated on each
host and uploaded; these are not GPU map-generation differences. This replay
audit uses the same atlas/build but is not a coordinate dump from the fresh
eight-step run itself.

Complete per-field maxima, worst points, identity differences, records,
checkpoint/export hashes and coordinate counts are preserved in
`docs/z4c_hyperboloidal/evidence/aurora-native-state-cross-platform.json` and
`build-native/aurora-native-state-cross-platform-v2/results.json`.

## Matched Aurora GPU control

The unchanged historical N12 GPU executable (SHA256
`494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753`)
was run fresh for eight full steps on the same two-node,24-rank configuration.
Only its stopping count changed to eight. Its pinned configuration, executable,
input, exact dt/time history and actual SP1 checkpoint metadata were verified.

The native and original GPU implementations **pass every short field gate**:

| State | Bulk max absolute error | Scri max absolute error | Failed field values |
| --- | ---: | ---: | ---: |
| Initial | 0 (bitwise identical) | 0 (bitwise identical) | 0 |
| Eight full steps | 5.5511151231e-17 | 4.3423780970e-17 | 0 |

Each state comparison covers all7,137,936 stored field scalars, including
dependent/dormant fields and evolved overlap. Eight-step bulk/scri weighted
RMS errors are4.1004829994e-20 and3.4372370898e-19. The largest ratios to the
unchanged tolerance are2.8481610858e-6 and4.3423725525e-5.
All688,176 actual bulk coordinate components match bitwise, as do the ten
resolved profile/gauge doubles serialized by both checkpoint identities.
The CPU-side exact identity/coordinate failures remain in the earlier audit;
the runtime restart identity guard has not been changed.

Every available original chi aggregate agrees exactly, including raw and
accepted minima and zero floor/nonpositive/nonfinite/correction counts.
Historical SP1 contains no per-cell chi ledger, so this matched-GPU result is
explicitly **aggregate-only for reference chi history**. Both native
checkpoints still contain the complete ledger. The original SP1's exported
all31 fields agree bitwise with its stored state; compensation arrays are
finite but are not misrepresented as chi history.

The matched-backend result separates the earlier CPU/Aurora reference mismatch
from a migration failure within this eight-step test. It does not replace
first-RHS,stage/poison,96-rank or long-evolution validation. Evidence is
`build-native/aurora-native-vs-original-gpu-eight-v3/results.json`, with a
preserved comparator source and dependency-provenance supplement; durable
copies are `docs/z4c_hyperboloidal/evidence/aurora-native-vs-original-gpu-eight*.json`.
Seven helper negative controls detect the last scri field, changed chi
aggregates and malformed coordinate membership. The original valid comparison
was not rerun for those controls.

`n12_checkpoint_gpu_control.py --ranks 24|96` supports the same matched-GPU
comparison on either requested decomposition. `n12_checkpoint_rank_compare.py`
compares complete native checkpoints across decompositions with **bitwise**
field/record gates, exact same-backend physics identity and optional actual
bulk-coordinate dumps. Its local format validation used the already captured
two/eight-rank writer fixtures; actual96-rank evidence remains a separate run.

The subsequent PBS9051435 eight-node/96-rank bundle passed the same matched
original-GPU comparison with no field failures, exact common resolved policy,
exact bulk coordinate bytes and exact available chi aggregates. Native24 versus
native96 is **bitwise identical** at both initial time and step8 across every
one of7,137,936 field scalars and1,842,048 ChiRecord scalars per snapshot. Actual
bulk cached geometry is also bitwise identical. Durable reports are
`evidence/aurora-native-vs-original-gpu96.json`,
`evidence/aurora-native24-vs96-initial.json` and
`evidence/aurora-native24-vs96-eight.json`. These complete-state correctness
results do not establish strong-scaling performance or long-run floor behavior.

## Separate strict Aurora sidecar

The CPU sidecar remains unchanged. `n12_checkpoint_map_portable.py` builds the
reviewed independent native/legacy map helper using the actual frozen Aurora
compiler/device/FP flags and native link objects. Its bounded one-rank run
performs no PDE evolution and must prove all bulk and scri coordinates bytewise.
`n12_checkpoint_bind_aurora.py` will create a new sidecar only after that proof
also matches actual native bulk replay coordinates and scri stage-trace
coordinates. It requires the admitted GPU24 template checkpoint, passed
matched-GPU result, matching frozen native executable provenance and independent
initial fields. It preserves exact CPU/GPU identity differences in the new
manifest and never relabels CPU surface coordinates as Aurora-exact. Until the
surface proof arrives, no strict Aurora sidecar is emitted.

## Common-time history readiness

`n12_long_history.py` compares the ordinary outputs from one or more segments
against all11 pinned N12 history rows through3551. It requires exact matching
checkpoint/history times, unchanged horizon settings, every full and harmonic
exterior RMS gate, adopted areal-radius/irreducible-mass gates and consistent
duplicates. It retains every duplicate and additional nominal-step output row.
The FastFlow summary's low-precision printed time is checked against its cycle
and the exact history time, without rounding the checkpoint/history clock.

Found/complete status is inferred from the reviewed runtime guard that must
pass before a history row is produced; ordinary files do not serialize status
flags or every mask bit. Complete finite harmonic coefficients and finite
positive horizon properties are checked, and that inference is explicit in
the report. Nine synthetic parser/gate controls pass in
`build-native/n12-long-history-controls-v2/results.json`; this is tooling
validation, not long-run evolution evidence. See
`tst/z4c_hyperboloidal/n12_long_history_contract.md` for scope and gates.

Remaining long-run evidence includes fresh complete native3551 trajectories,
their11 common-time histories and active chi-floor behavior. The original
target has56664 recorded floor events in eight cells, but its ledger is only
available as aggregate diagnostics. Counts/corrections/first-floor differences
must be reported rather than replaced with fabricated per-cell references.


## Exact Aurora sidecar

The separate strict Aurora sidecar is
`build-native/n12-checkpoint-map-aurora-v1`, mapping SHA256
`484bd5e77d83c537ce564e2f02b68bd3544680ced65d81abdf8a69216434b15d`.
The CPU sidecar and all cross-platform failures remain unchanged. This sidecar
binds the actual Aurora resolved physics identity and admitted original-GPU
initial checkpoint to independently checked topology and complete point order.
Every one of688176 bulk and2592 scri coordinate scalars agrees bitwise between
the independent Aurora helper and actual native geometry. The helper run has
an explicit successful execution record and post-teardown PASS marker; the
failed, complete-looking files from job9051466 are not admitted.

An extra initial-state bitwise check exposed a phase distinction and remains
recorded as failed. The geometry helper copies raw host
`SphericalTransfer3D` constructor reference fields; native initialization then
calls `EnforceAlgebraicConstraints` and `SurfaceExchange::Project` in
`InitializeBoundaryValues`. Comparing these different phases gives182656 bulk
Azz signed-zero differences, with zero numerical error. Scri has2904 bit
differences:2346 signed zeros and558 numerical differences, at most
2.220446049250313e-16 in lapse/shift. The extra constructor check therefore uses
the already adopted short/chi gates, with zero failing values and explicit
`bitwise_equal:false`. The exact geometry, topology, identity and unchanged
original-GPU accepted-initial-state gates remain exact; no numerical tolerance
or runtime code changed. Complete per-field differences and the preserved
extra-check failure are hash-bound in the sidecar proof.

Strict native exports now pass for actual initial24 and step8 at24/96 ranks
using this Aurora identity. The eight-step references remain the immutable
original GPU files, with separate metadata linking their unchanged bytes to
the new map. All31 fields and all8 native records are exported. Comparison of
these old control records remains aggregate-only; the independent GPU stage
accessor provides the separate full per-cell ledger proof. This does not
supply historical step3551 per-cell records or long-evolution evidence.
