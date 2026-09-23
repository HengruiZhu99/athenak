# Mesh-only cleanup regression

`MeshBlockPack` owns coordinates created only by `AddCoordinatesAndPhysics`.
The `-m` branch in `main.cpp` deletes the mesh before that function runs.
Previously the pack constructor left `pcoord` indeterminate and the destructor
executed `delete pcoord`; this is undefined behavior on mesh-only exit. The
constructor now initializes `pcoord` and `pmb` to `nullptr`. Normal construction
still overwrites both before use. No equations, mesh criteria or physics change.

`test_constructor.cpp` placement-constructs the real pack in memory prefilled
with 0xa5 and inspects both pointer object representations, without evaluating
an indeterminate pointer. It fails against the original object, sets invalid
members to null for safe test cleanup, and passes with the corrected constructor.
The null-pointer representation comparison is intended for the supported x86-64
and arm64 platforms. A passed test does not certify a stellar evolution.

`build_regression.py` takes an existing CMake Unix Makefiles build and a corrected
`meshblock_pack.cpp`. It recompiles only that translation unit, links the
regression against both old and corrected objects, and creates a separate
Athena executable. It preserves original compile/link flags and reuses all other
objects/libraries, records their hashes and commands, and checks that the old
build was not modified. It requires an unused output directory and expects the
original constructor regression to fail; it is specifically a before/after
regression for this repair.

Example:

```sh
python3 analysis/mesh_probe_cleanup/build_regression.py \
  --build /absolute/path/to/original-build \
  --fixed-source /absolute/path/to/src/mesh/meshblock_pack.cpp \
  --output /absolute/path/to/new-build
```

Local validation also exercises `-m` on the complete 988-block stellar mesh with
one/four MPI ranks. These mesh-only tests raise `max_nmb_per_rank` to1000 to fit
the mesh on fewer ranks; this is not an evolution test or a production input
change. Three-cycle vacuum and nonzero-lapse controls on four ranks compare
83 binary snapshots/history files per case byte for byte against the original
executable. The full516-rank GPU mesh probe remains required in the allocation.

The defect is consistent with job8855874's rank391 SIGSEGV during `-m` cleanup.
Without a backtrace, this is not an identification of that job's exact faulting
instruction. Stellar evolution never started in that failed job.
