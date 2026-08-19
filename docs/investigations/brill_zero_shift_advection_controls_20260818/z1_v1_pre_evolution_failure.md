# Z1 v1 pre-evolution failure

The first dedicated one-GPU `shared_interactive` attempt, Slurm job `57250327`,
configured and built the CUDA/MPI executable and passed all five focused tests.
AthenaK then imported the initial data but stopped before the first PDE step with
`AMR history: history source-id mismatch`. The immutable N128 hierarchy authority
was recorded by `ac75c8d348da91b38cbc6855b5fba51cd3089663`, while the new shift-control
source necessarily has a different commit identity. No evolution data were
produced.

This is classified as a fail-closed replay-provenance harness blocker. The v2
successor adds an optional replay-only compatibility declaration that must equal
the recorded header source ID exactly; default source binding and all tree,
geometry, schedule, checksum, and restart checks remain unchanged.

- Remote root: `/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-shift-controls-90838417-v1-20260818`
- Terminal manifest SHA256: `583601841c22e76de5ee6c1e66068e75415f916bfc9941c6bde2a1e2c8fc1d83`
- Focused-test log SHA256: `b1bdda211adf89fe5cad0661da0b9b39a5c262cbf6cef1c0e9b5c18933d90d10`
- Run log SHA256: `5a6e3b3793548c2a6f5c5200d4b634d3d7252755f84eb4dbc9793690df41b87a`
- Executable SHA256: `7a6575b16e9ad9b2623d17a27b4e89b36423d866164bd3e93fd5d85ca82e5062`
- CMake cache SHA256: `45dccecabb611ab0b539116730cefaae57e66a8e8f361405004e8cb7efba2067`

The remote `terminal.sha256` manifest verifies in full.
