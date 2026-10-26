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

- Five automated tests pass: polynomial radial operators; nonlinear manufactured
  shift; independent FD elliptic boundary-value solve; nonlinear tensor-kernel
  metric-connection damping and scalar-Theta source audit; unsubtracted Minkowski
  residual. They do not complete the full equation audit.
- Finest manufactured shift error: 3.77e-15; independent BVP discrepancy: 2.00e-14.
- Raw actual-scri l'Hopital endpoint candidate **fails**: N64 Minkowski and a
  1e-3 lapse pulse lose positivity at t=0.0598425, CFL=.05. N32 reaches t=.1 but
  already violates the pole and drift gates (pole max~1.36, field drift~.302).
  Survival to .1 was not a pass. Endpoint compatibility/discretization needs repair.
- Aurora SSH failed. User subsequently authorized local one-dimensional testing;
  that instruction supersedes the initial execution-resource restriction for this work.
- No published experiment has yet been reproduced. No coupled scalar, trumpet,
  wormhole-to-trumpet, long-duration, convergence, or restart acceptance is complete.

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
