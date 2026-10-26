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
Twenty-five CTest groups pass. Nonlinear physical ADM, native tensor, scalar,
inner/scri limit, Bondi-frame, compiled/Python and SBP algebra audits are included.
The small .001 gauge pulse reaches t=10,000 at N256 and1000 at N1024;
matching finer durations are in progress. Finite-amplitude scri peaks still fail
acceptance. Strong coupled dispersal has three resolutions and approximately
fifth-order mass-budget convergence; its finest final discrepancy is8.2e-9.
Full principal time/elliptic/dissipation studies and long scalar evolutions remain
unfinished. No trumpet, wormhole-to-trumpet, collapse, or literature reproduction
acceptance is complete.

The new isotropic wormhole construction and separate inner-end analysis are in
`puncture-asymptotics.md`. An evolved AF coefficient endpoint replaces the old
finite-cutoff outflow assumption. Its nonlinear runs still fail constraint
convergence. Gauge/damping/source/SBP controls and a separate incoming-gauge
boundary treatment preserve all failures; none is an admitted cure.

## Reproducible build and tests

From repository root (C++17 compiler, CMake, Python3+NumPy):

```sh
git submodule update --init --depth 1 kokkos
cmake -S tst/elliptic_shift_1d -B build-elliptic-shift-1d -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-elliptic-shift-1d --target spherical_z4c spherical_fast scri_chart_probe -j 2
python3 tst/elliptic_shift_1d/test_elliptic.py --library build-elliptic-shift-1d/libspherical_z4c.dylib
ctest --test-dir build-elliptic-shift-1d --output-on-failure
```

Linux uses `libspherical_z4c.so`. To reproduce the failed endpoint pilot, choose
an unused output directory:

```sh
python3 tst/elliptic_shift_1d/evolve.py --library build-elliptic-shift-1d/libspherical_z4c.dylib --n 64 --t 1 --cfl .05 --output output/elliptic-shift-1d/reproduce-raw-endpoint
```

Outputs include histories, profiles, failure receipts, source/library hashes,
commands, and the last completed-step checkpoint. These outputs are intentionally
excluded from Git. Keep them in place; concise summaries belong in evidence/.
