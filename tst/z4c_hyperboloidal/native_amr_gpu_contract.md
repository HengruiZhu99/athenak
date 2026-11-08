# Bounded coherent-candidate GPU AMR tool contract

These tools prepare the seven-launch set in
`docs/z4c_hyperboloidal/gpu-amr-readiness.md`. They never submit jobs. The reviewed
admission patch remains isolated; the fixed-SMR N12 source/build/executable are
unchanged. Every new AMR input uses independently selected fixed dt `.0002`.
The preserved larger-dt tests did not establish predecessor admission parity.
No numerical or comparison tolerance is changed.

## Candidate build

`native_amr_admission_candidate.py package --root <frozen-source>
--reviewed-overlay <clean-overlay> --output <new-package>` copies the exact
five reviewed files, patch, and existing CPU prototype proof, binding every
byte. Its old three-object `build` mode is retained for prototype provenance;
**that mixed-object executable cannot qualify the GPU candidate** because
other baseline objects saw a different class declaration.

Use `native_amr_candidate_cmake.py build --baseline-build <frozen-build>
--package <package> --output <new-directory>` for the final candidate. It copies
source physically into a separate tree, applies only the exact five-file patch,
replays the baseline's resolved CMake options, and builds the ordinary target
with `make -j64 athena`. Exporting compile commands and changing directory paths
are the only configuration differences. Every native compile recipe must match
after path/dependency-output normalization. Link flags, libraries and their
order must also match after path relocation. The original main source is exact;
all translation units and Kokkos objects are compiled coherently. No baseline
build object is linked. `verify --output <directory>` rechecks the source,
package, dependencies, object/library and executable hashes. The new candidate
SHA is never described as the frozen N12 executable SHA.

`native_amr_gpu.py build --case lifecycle|evolution --build <candidate>/build
--candidate-build <candidate>/provenance.json --output <new-directory>` takes
actual candidate diagnostics.cpp compiler/device flags and replaces only its
main object with the existing test main. No runtime overlay is applied at
this step. It records `native-amr-gpu-build-v3`, the coherent candidate build
hash, candidate normal-main SHA, distinct baseline SHA, test source/object and
all linked/dependent inputs. A helper for the former mixed-object candidate
has v2 provenance and is explicitly rejected by GPU qualification.

## Inputs and seven launches

`native_amr_gpu_input.py --output <new-directory>` produces:

- `amr_helper.athinput`: original initialization fixture plus only `dt=.0002`.
  Pass it to lifecycle with `--diagnostics` and to evolution without options.
- `amr_restart.athinput`: the same dt and original model, ordinary dynamic
  classical RK4, location criterion `(.99,0,0), rad=.005`, capacity16, and
  shared restart output each cycle.

Run lifecycle and evolution once each on24 and96 ranks. Run the coherent
ordinary executable for a24-rank prefix throughcycle2, a96-rank continuous run
throughcycle4, and a96-rank `-r` resume of the verified prefix cycle2 checkpoint
throughcycle4. Resume supplies neither input nor dt override. Use the existing
Aurora tile launcher and preserve every duplicate checkpoint. No command here
constitutes submission authorization.

## Unchanged scientific gates

`native_amr_gpu_compare.py --dt .0002` checks:

- `--case lifecycle --run 24=<folder> --run 96=<folder>`: actual106→127→106 tree,
  all229392→277236→229392 bulk and864→1296→864 surface points, seeded nonzero
  history, unchanged migrated leaves, polynomial gates and diagnostics freshness.
- `--case evolution --run 24=<folder> --run 96=<folder>`: all five accepted states,
  four actual Driver cycles,106→127→127→106→106 topology,16 poisoned RHS stages,
  exact stage times, coverage, compatibility≤1e-7 and finite clocks/CFL.
- `--case restart --prefix <folder> --continuous <folder> --resumed <folder>`:
  ordinary adaptive input,113-leaf outer refinement, all meaningful metadata,
  topology/unit costs, exact time/dt, all padded31-field/eight-record scalars.
  Prefix24 cycle2 equals continuous96 cycle2; resumed96 cycles3/4 equal their
  continuous counterparts. Repeated outputs must also agree.

Independently enumerate logical keys and bounds; reject missing/duplicate/extra
or non-owned points. All31+8 comparisons remain bitwise. History reductions are
minima for0/1/7, maximum for5 and sums for2/3/4/6. Zero diagnostic volume never
removes an evolved point. Polynomial tolerance remains256ε×32768×.02; C++
constant tolerance remains512ε. Compare initialized restart fixed-header slices
`[4:120]+[156:-20]`; record the nine pre-existing unused coarse integers
separately. No meaningful metadata gate is removed.

## Execution provenance

GPU comparison requires `--candidate-package <locally-reviewed-package>` and
`--execution-evidence <json>` with schema `native-amr-gpu-execution-v2`:
`case`, `candidate:true`, `exit_status:0`, `baseline_inputs_unchanged:true`,
`fixed_dt:.0002`, separate `baseline_native_executable_sha256` and
`candidate_executable_sha256`, and `candidate_build_provenance` pointing to the
coherent build record. `candidate_compile_database` and `candidate_link_file`
point to the actual hash-bound CMake files from that build. Verification also
rehashes those files and any read-only finalizer script. `artifacts` maps every consumed relative path to SHA256;
all referenced build records must be included. Snapshot cases additionally
supply `build_provenance` for their v3 helper. The exact reviewed package/patch
must bind that candidate, and actual SYCL compile recipes must cover all native
translation units. Baseline/v1 and mixed-object/v2 helpers are rejected.

`executions` has exactly labels `24`,`96` for snapshot cases or
`prefix`,`continuous`,`resumed` for restart. Each contains `directory`, `ranks`,
`exit_status`, actual `executable_sha256`. Each fresh run also records the exact
generated `input_sha256`; resume records `input_override:false` and
`dt_override:false`. The collector obtains actual command/scheduler status and
hashes from completed runs; output-looking files alone prove nothing.

`--cpu-fixture-evidence` performs offline reader validation only and cannot
supply GPU evidence. Prior original-dt fixtures require their explicit `--dt`;
clean candidate prototype checkpoints require `.0002`, capacity72, prefix2,
continuous2 and target8. Existing prototype restart parsing is not a coherent
candidate restart proof. The local controls additionally reject wrong patch,
old dt, mixed-object build, baseline-as-candidate, wrong class definition,
CPU-as-GPU and modified package bytes.

These bounded tests do not establish long-time adaptive stability, AMR
convergence, agreement with an independent adaptive solution or performance.
Fixed-SMR N12 long-run admission remains separate.
