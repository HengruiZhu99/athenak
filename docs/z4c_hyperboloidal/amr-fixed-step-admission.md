# Fixed-step AMR admission candidate

This is an isolated, reviewed source overlay. It has not been merged into the
native runtime. Its bounded GPU transfer, evolution and restart qualification
now passes on24/96 Aurora ranks at fixed dt `.0002`. Its coherent ordinary
executable and both AMR helpers have compiled successfully with the actual
Aurora SYCL toolchain in separate source/build directories. The frozen N12 source, objects,
executable and prior unsafe-dt consistency evidence remain unchanged.

The available predecessor AMR driver checks a fixed requested step at fresh
production initialization, restart, and after actual regridding. Its cap is
`min(requested_dt, target / characteristic_rate, .005)`, with fresh configured
target .2 and continuation target .4; rejection uses the existing relative
margin `1e-12`. Its separate fresh nonproduction path can choose a smaller step
once. This candidate implements fixed-requested admission, not that automatic
selection path or a new adaptive controller. The fresh target comes from the
native `time/cfl_number`, which is .2 in the pinned input. The absent optional
angular-dissipation operator is not emulated.

The proposed method runs only on adaptive meshes. Driver initialization passes
its actual restart flag, including for a cycle-zero restart. MeshRefinement
passes continuation=true after boundary preparation on an actual new mesh.
Fixed SMR returns before any characteristic kernel or reduction. At both call
sites the current residual and solved endpoint normal correction are available.
The method is explicitly restricted to those prepared states; stage-four RHS
workspace is not an accepted-state cache.

The kernel reads central restored values and the existing normal-lift value,
then evaluates metric inverse and the existing characteristic formulas. It does
not evaluate derivatives, a bulk or surface RHS, KO, a new boundary exchange,
or a normal solve. Host preflight and point failures are collective before an
exception. Each physical/speed component is checked for finiteness before fmax
can conceal a NaN. MPI calls are guarded for serial builds. Test callbacks,
per-point exports and prototype logging have been removed.

The preceding prototype compared every initial, refined, and coarsened point
against the unchanged full RHS and obtained exact speeds. It rejected unsafe
fresh, continuation and post-refinement steps. Independently chosen fixed dt
.0002 completed four cycles with 106→127→106 blocks at two and eight ranks;
all owned 31-field/eight-record values matched the unchanged runtime exactly.
Those earlier reports and the original larger-dt consistency fixtures are
preserved separately.

The clean patch prototype was built with make -j64 and the original main object,
replacing three runtime objects. This is **mixed-object prototype numerical
evidence**, not a coherent production build: the patch also adds a member
declaration to `Z4cHyperboloidal`, so unchanged class layout alone does not remove
the different-class-definition/ODR concern in other baseline translation units.
These prior results are preserved rather than silently upgraded. A real
ordinary adaptive input at dt .0002 refined the outer +X parent (106→113 blocks).
Four continuous cycles on two ranks matched a two-cycle prefix on two ranks
followed by ordinary `-r` checkpoint deserialization on eight ranks. The resumed
command supplied no input file or timestep override. Complete checkpoint data
at cycles three and four matched bitwise, including all 26,336,232 padded field
and history scalars, initialized metadata, topology, and unit block costs.
Duplicate checkpoints were checked rather than discarded. Nine native unused
coarse-index integers are recorded separately, as in the existing restart
protocol. All other meaningful metadata is exact. A separate MPI-disabled
compile of the lifecycle succeeded and has no unresolved MPI symbols; a serial
runtime evolution was not tested.

Final candidate qualification requires a separate complete CMake source/build
tree. `native_amr_candidate_cmake.py` physically copies the source, applies the
exact five-file reviewed overlay, and compiles the ordinary target with
`make -j64 athena`. It preserves original main source and verifies every native
translation unit's normalized compiler command against the frozen build.
Candidate objects and Kokkos libraries are rebuilt together; no baseline-built
object is linked. Source, patch, recipe, object, executable and baseline hashes
remain distinct. The baseline source/build are never patched or rebuilt.

The coherent local build is in `build-native/amr-admission-coherent-local-v1`.
Its initial post-build audit mistakenly selected duplicate unit-test recipes
for three sources; `pre-finalization-failure.json` preserves that attempt.
The corrected audit selects only `athena.dir` recipes and confirms all 198
native compilation recipes match. This audit correction did not rebuild or
change numerical objects. The peer-reviewed `audited-provenance.json` also
enforces exact normalized link flags/library order and binds the actual CMake
compile database, link file and finalizer bytes. Earlier audit records remain.
A targeted actual four-cycle ordinary-main endpoint comparison passed in
`endpoint-proof.json`, with all26,336,232 padded field/history scalars exact;
a separate two-rank manufactured-state lifecycle preflight passed refinement,
coarsening and diagnostic freshness at the safe dt. The earlier
two-to-eight-rank restart evidence remains prototype evidence. Separate actual
GPU coherent transfer/evolution/restart qualification is recorded below.

The actual Aurora build is recorded in
`evidence/native-amr-coherent-aurora-build.json`: all198 native translation-unit
recipes and the full link policy matched the baseline, and its source/objects
were verified unchanged after every build. This is compilation/provenance
evidence only. The later seven dt=.0002 GPU validation launches completed
normally in jobs9051631 and9051645, and all three primary scientific comparisons
pass. See `evidence/native-amr-coherent-gpu-validation.json`.

The actual lifecycle checks106→127→106 on24/96 ranks, nonzero history
redistribution, all31+8 complete states, exact unchanged-leaf migration and
fresh/read-only diagnostics. Four actual Driver cycles through two regrids
produce five bitwise-equal complete states and16 valid poisoned-workspace stages
per decomposition. Maximum compatibility is1.0910657402209053e-15; clock and CFL
are finite, with maxima1.822919687174616e-4 and.20312566084213138. These stage CFL
values are observations, not a new per-stage admission threshold.

The coherent ordinary-main113-leaf case now proves actual24→96-rank checkpoint
deserialization: prefix cycle2 and resumed cycles3/4 agree bitwise with the
continuous96-rank run across all26336232 padded field/history scalars. Exact
initialized metadata, time/dt, topology and unit costs pass, including duplicate
outputs. Resume uses no input or dt override. The inherited nine unused global
coarse-index integers remain reported separately under the unchanged contract.
This validates bounded consistency and restart behavior of the isolated
candidate; it does not establish adaptive convergence, long-time stability,
independent adaptive-solution agreement or performance. Runtime integration
remains pending while the frozen N12 trajectory through3551 is completed.

Review patch: [native-amr-fixed-step-candidate.patch](evidence/native-amr-fixed-step-candidate.patch),
a byte-identical retained copy of `build-native/amr-admission-clean-v1/production.patch`
(SHA-256 `9f411610ee3246f39a25be3e9f19598186da8c28bdeb90c8f53bc57489ebc2d7`).
Full ordinary-restart evidence: `build-native/amr-admission-clean-v1/restart-proof/results.json`.
Preserved prototype summary: `evidence/native-amr-clean-admission-cpu-proof.json`.
The parent task controls later runtime integration; the frozen N12 source,
objects and executable remain unchanged.
