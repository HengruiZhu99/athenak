# Spherical elliptic-shift experiment

**Active and incomplete.** Worktree branch: `codex/hyperboloidal-elliptic-shift-1d`.
Baseline snapshot: `f5d99045e0168e844f87fc0b6189593aedd18da9`, separate from new work.
Original checkout remains on `codex/cubed-sphere-boundary` with its changes untouched.
All src/tst/scripts/inputs dependencies and current notes/goals were preserved.
Large historical generated outputs remain in the original checkout; their excluded
paths are inventoried in `evidence/baseline-preservation.json`. The Kokkos gitlink
was initialized at its exact recorded revision in this worktree.

The source notes are copied verbatim under `specification/` with SHA-256 provenance.
They claim a linear compatible graph-phase result only. No nonlinear or black-hole
stability is inferred. See `nonlinear-formulation.md` for new independent derivations,
`acceptance.json` for gates, and `objective.txt` for the complete user requirements.

## Current evidence

See [the active scientific work record](progress.md) for passing component checks,
resolved failures, source/checkpoint ancestry and the next derivation steps.
The two CTest groups pass. The independent scalar oracle passes its declared
preliminary convergence/error gate, and the wormhole initial-data constraints
are verified. The nonlinear actual-scri pulse evolution remains **unaccepted**:
resolved runs lose boundary compatibility as the pulse reaches scri. No coupled
scalar, trumpet evolution, wormhole-to-trumpet, long-duration or literature
reproduction acceptance is complete.

## Reproducible build and tests

From repository root (C++17 compiler, CMake, Python3+NumPy):

```sh
git submodule update --init --depth 1 kokkos
cmake -S tst/elliptic_shift_1d -B build-elliptic-shift-1d -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-elliptic-shift-1d --target spherical_z4c -j 2
python3 tst/elliptic_shift_1d/test_elliptic.py --library build-elliptic-shift-1d/libspherical_z4c.dylib
```

Linux uses `libspherical_z4c.so`. To reproduce the failed endpoint pilot, choose
an unused output directory:

```sh
python3 tst/elliptic_shift_1d/evolve.py --library build-elliptic-shift-1d/libspherical_z4c.dylib --n 64 --t 1 --cfl .05 --output output/elliptic-shift-1d/reproduce-raw-endpoint
```

Outputs include histories, profiles, failure receipts, source/library hashes,
commands, and the last completed-step checkpoint. These outputs are intentionally
excluded from Git. Keep them in place; concise summaries belong in evidence/.
