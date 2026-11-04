# Promote direct halo setup and simplify solver lifecycle code

`hyperbolic_shift_3d_amr_evolution` now uses the validated direct-construction
path by default. The standalone and coordinated plan builders pass the final
inner-transition policy into `HyperbolicComposite`; neither constructs and then
replaces an unused halo plan. The general setter and old/new contract test remain
available for explicit policy changes and regression testing.

Removed the `HYPERBOLIC_DIRECT_HALO_SETUP` and
`HYPERBOLIC_COMPACT_DIAGNOSTIC_TRANSFER` compile-time experiment switches, the duplicate
`hyperbolic_shift_3d_amr_setup` executable, the unsuccessful diagnostic-transfer
executable, and its unused snapshot class and component test. Diagnostic reads
keep the measured original transfer path, including complete padded blocks for
moving-horizon sampling. AMR CMake target setup is shared in a small helper.

The AMR driver, shared MPI evolution, profiling, and coordinated-plan lifecycle
files are consistently formatted with expanded control flow. The mesh selector
and diagnostic download use named intermediate values instead of nested dense
expressions. Include ordering, arithmetic order, scientific configuration,
checkpoint identity and acceptance/capture semantics are preserved.

The earlier [performance report](../performance-scaling-v1/README.md) remains an
immutable historical measurement. This cleanup introduces no new performance
claim. It makes that validated setup optimization the default and removes the
unhelpful implementation alternatives. Independent scientific variants and
reproduction evidence are retained.

Validation uses the frozen optimized executable/results from `71765458a` as the
AMR reference, and the frozen pre-promotion V11 fast executable for the sparse
integration comparison. Commands and source/binary hashes accompany the results.
No new Aurora allocation is needed for this promotion; its GPU setup mechanism
was qualified in the previous experiment. The current changes are checked locally
with Serial Kokkos and MPI, including the coordinated builder's lifecycle.

## Validation evidence

The local AMR matrix covers one- and four-rank dynamic waves (including poisoned
unused workspace), two-rank Minkowski, uniform wave/boost, fixed boost, the known
dynamic-boost rejection, and a two-rank save followed by a four-rank restart.
All nine cases passed. All 20 full checkpoint, history, and rejection-artifact
comparisons match the frozen optimized references byte for byte. All six
malformed or incompatible checkpoint cases reject collectively as expected.

The source and compiler-definition record confirms that the default target has
the optimized target's scientific definitions; only the now-unnecessary
`HYPERBOLIC_DIRECT_HALO_SETUP=1` definition is removed. Formatting checks and
Python syntax checks pass. The CMake target inventory includes the default
AMR executable and excludes all three retired experimental/component targets.

Both CTest contracts passed: coordinated mesh-plan publication and exact
old/new setup equality across 8-, 15-, 71-, and 22-leaf topologies. The two-rank
transaction test passed invalid-repartition recovery, preservation of accepted
fields and nonzero RK compensation, checkpoint writing after recovery, changed
ownership followed by continued evolution, and exact rollback/failure capture.

The poisoned sparse-MPI integration suite also passed: pulse trajectories across
rank partitions and restart, Minkowski and puncture checkpoints, and deliberately
failed RK-stage rollback/capture all match the frozen V11 fast control bitwise.

The local wall times are correctness-run metadata, not a new performance
benchmark. The previously measured CPU/GPU improvements remain qualified by
the workloads and baselines in the historical performance report. No GPU
rebuild or scaling sweep was performed for this cleanup.

* [Validation commands](evidence/commands.json) and [completion record](evidence/completed.json).
* [Twenty bytewise comparisons](evidence/bitwise.json).
* [Source/compiler bindings](evidence/source-and-definitions.json) and
  [binary/output hashes](evidence/artifacts.json).
* [Raw logs and histories](evidence/raw-text-evidence.tar.gz), including the
  successful CTest details, checkpoint-rejection manifest, and sparse verification.
* [Package hashes](evidence/SHA256SUMS). Large checkpoint and executable payloads
  remain in `build-default-cleanup/` with their hashes recorded; they are not
  added to Git.

The exact local orchestration is preserved in [validate.py](validate.py). It
expects the frozen reference artifacts in `build-performance/`, the rebuilt
current binaries, and a fresh `build-default-cleanup/evidence/` directory.
The evidence archive also includes the packaging script and test runner used.
