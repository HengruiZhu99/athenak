# Bounded GPU AMR candidate qualification

The AMR admission patch is reviewed and remains isolated. The frozen fixed-SMR
N12 source/build/executable and all older successful/failed evidence remain
unchanged. This preparation does not submit GPU work and does not block the
separate fixed-SMR N12 baseline comparison.

A source audit found a genuine carryover gap: the predecessor checks a fixed
requested step after initialization/restart and real refinement. The original
dt can fail that check on the refined fixture. New candidate tests therefore
use independently chosen `.0002`; this is an explicit new fixture, not a
relaxed gate or a revised claim about old larger-dt consistency tests.
See `amr-fixed-step-admission.md` for the exact cap and cadence.

The old clean overlay compiled only three replacement objects. Its exact
value-only/full-RHS oracle, safe four-cycle tests and genuine2→8 restart provide
prototype numerical evidence. They do not establish a coherent executable:
adding a member declaration means other baseline TUs saw a different class
definition, even with unchanged layout. Preserve those results as prototype
proof and reject them as the final GPU build.

The final candidate uses a separate physical source copy and ordinary full
CMake build, applying exactly the reviewed five-file patch. All198 native TUs
and Kokkos objects compile coherently; original main source and every native
compile recipe remain unchanged after directory normalization. Baseline files
and objects are not edited/rebuilt or linked into the candidate. Distinct
baseline, package/patch, coherent candidate and helper executable hashes are
required. `native_amr_candidate_cmake.py` implements and verifies this route;
`native_amr_gpu.py` builds the existing lifecycle/evolution mains against its
coherent objects. The original main is used for the ordinary restart case.

The complete bounded GPU set has passed all seven MPI launches:

| Case | 24 ranks | 96 ranks | Required evidence |
| --- | --- | --- | --- |
| lifecycle `--diagnostics` | once | once |106→127→106 transport, complete31+8, nonzero history redistribution, migrated unchanged leaves, read-only/fresh diagnostics |
| evolution | four cycles | four cycles |106→127→127→106→106,16 poisoned RHS stages, exact clocks/coverage, accepted criterion halos, current plans and all-one costs |
| coherent ordinary executable | prefix throughcycle2 | continuous throughcycle4; resume24-rank prefix throughcycle4 | normal adaptive input and actual checkpoint deserialization,113-leaf outer-refined fixture |

All inputs use fixed dt `.0002`. Generated helper input changes only dt in the
original initialization fixture; the C++ helpers retain their own criteria and
static construction followed by native dynamic Driver setup. Their capacity is
`160/ranks+8`. Ordinary input uses location(.99,0,0),radius.005 and capacity16,
sufficient for113 leaves on24/96. Ordinary resume supplies no input/dt override.

For snapshots independently enumerate all leaves and owned bounds, including
zero-volume overlap and surface children. Compare all31 fields and8 ChiRecords
bitwise across24/96, with exact history reducers. For restart compare24-rank
prefix cycle2 with96-rank continuous cycle2 and resumed cycles3/4 with continuous
counterparts. Include padding, exact physics/atlas identity, topology/unit costs
and time/dt. Check all duplicate files. As before, nine unused coarse-index
integers are reported separately from initialized fixed-header slices
`[4:120]+[156:-20]`. No scientific tolerance is altered.

`native_amr_gpu_contract.md` specifies v2 execution evidence and v3 coherent
helper provenance. Each collector must bind actual command exits, launcher/
scheduler artifacts, SYCL compilation, distinct executable identities and all
consumed files. A CPU build, mixed-object build, frozen baseline executable or
old-timestep input must fail candidate admission. Portable tools never submit.

Local preparation evidence is in `build-native/amr-admission-coherent-local-v1`
and the separate candidate parser/control directories. This is CPU evidence
only. The earlier offline2/8 controls and larger-dt consistency results remain
untouched. The isolated coherent SYCL build and both helper builds have now
passed on Aurora. The seven actual GPU runs and their three scientific
comparisons now pass at dt `.0002`. Neither these
bounded tests nor their CPU equivalents establish adaptive convergence,
long-time stability, independent-solution agreement or performance.

Final review package: `build-native/amr-admission-gpu-ready-v2/setup.json`.
It includes exact build/helper/launch/comparison command arrays and26 hashed
package/tool/input files. Its original GPU-built/submitted/evidence flags remain
false as a preserved preparation snapshot; actual execution is recorded below.
The peer-reviewed local build audit binds compile/link recipe files and their
exact normalized policy. The targeted coherent ordinary endpoint and
manufactured lifecycle checks passed;12 provenance controls passed. Durable
scope and hashes: `evidence/native-amr-coherent-candidate-preparation.json`.

Actual Aurora build evidence is separate:
`evidence/native-amr-coherent-aurora-build.json`. The ordinary executable is
`442594b847ff0eb56407513da48e96930ec41dbe537a69c202fd802da4ea6a31`;
it is not the frozen N12 executable. All198 native compile recipes and link
flags/library order match the baseline after path relocation. Compiler-version
and MPI-wrapper records match bytewise. All1972 baseline source files and228
objects passed checks before and after the three builds. The exact reviewed
patch remains isolated. All68 compact downloaded evidence files passed hash and
size verification; native objects/executables remain on Aurora.

Actual execution and comparison evidence is now recorded separately in
`evidence/native-amr-coherent-gpu-validation.json`. Jobs9051631 and9051645
completed normally; all453 two-node and1615 eight-node archived files passed
size/hash verification. Lifecycle states through106→127→106 match bitwise
across24/96 ranks for all31 fields and8 records. Nonzero history totals are
preserved,103 unchanged leaves remain exact through each regrid, and diagnostics
match fresh workspaces while preserving state at all three phases. The refined
snapshot has277236 bulk and1296 surface points. Polynomial errors remain below
the unchanged gate (maximum1.3010426069826053e-17).

Four actual Driver cycles through106→127→127→106→106 produce five bitwise-equal
complete states and16 valid poisoned-workspace stages per decomposition.
Maximum compatibility is1.0910657402209053e-15; finite clock/CFL maxima are
1.822919687174616e-4 and.20312566084213138. The ordinary adaptive113-leaf case
uses the coherent original-main executable. Prefix24 cycle2 and resumed96
cycles3/4 match their continuous96 counterparts across all26336232 padded
field/history scalars, exact initialized metadata, time/dt, topology and unit
costs. All duplicate outputs are checked; resume supplies no input/dt override.

This qualifies the isolated candidate for those bounded GPU cases. Runtime
integration remains pending while the frozen fixed-SMR N12 trajectory is
preserved through3551. No independent adaptive-solution comparison, adaptive
convergence, long-time stability or performance is established.
