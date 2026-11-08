# Bounded GPU AMR candidate readiness

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

The smallest complete GPU set still requires seven MPI launches:

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
untouched. Actual coherent GPU build, both decompositions' transfer/coupled
runs, and repartitioned ordinary restart remain outstanding. Neither these
bounded tests nor their CPU equivalents establish adaptive convergence,
long-time stability, independent-solution agreement or performance.

Final review package: `build-native/amr-admission-gpu-ready-v2/setup.json`.
It includes exact build/helper/launch/comparison command arrays and26 hashed
package/tool/input files, with GPU-built/submitted/evidence flags all false.
The peer-reviewed local build audit binds compile/link recipe files and their
exact normalized policy. The targeted coherent ordinary endpoint and
manufactured lifecycle checks passed;12 provenance controls passed. Durable
scope and hashes: `evidence/native-amr-coherent-candidate-preparation.json`.
