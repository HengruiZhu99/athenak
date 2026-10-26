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
Thirty-two CTest groups pass, including nonlinear tensor/ADM, scalar, scri,
inner-end, positive-field, Bondi and compiled/Python audits. The positive-variable
finite .1 gauge pulse reaches t=6 at N512/1024/2048. Physical-field and accepted
constraint convergence are near fifth order; finest boundary/affinity gates pass.
Independent endpoint quadrature isolation still fails, so this is **not an
admitted principal result**. An analytic-limit control is in progress. See
`positive-variables.md` for variables, source changes and diagnostic coverage.

The small .001 pulse completes t=10,000 at N256/N512 and3000 at N1024;
N1024 is continuing to10,000 from its exact frozen checkpoint. These older runs
have sampled diagnostics. Strong coupled dispersal has three resolutions and
approximately fifth-order mass-budget convergence, with finest discrepancy
8.2e-9 at t=6. Long scalar evolutions and its principal controls remain unfinished.
No successful collapse, wormhole-to-trumpet or full literature reproduction is
claimed. Already-trumpet constructions and labeled excision/source controls are
implemented, but their longer runs still fail an inner characteristic guard.

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
