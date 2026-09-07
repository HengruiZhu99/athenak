# Clean-reduction checks

Work in progress. `src/pc_gh/commuting_transfer.hpp` implements only the local
residual transfer algebra, with explicit source/destination reconstruction
targets. It is not yet connected to AthenaK communications. In particular this
does not supply the missing six-cell primary support for FD6 derivative ghosts.

`transfer_probe.cpp` exposes that primitive for CPU matrix fixtures. It is not
a production AMR or CUDA check. Promotion remains blocked until complete
halo/topology, restriction, prolongation and postprojection exchange tests pass.

The pinned mathematical suite is run from an external isolated copy. See the
evidence directory for the original suite output and scope. No optional GLM or
common-energy optimization is part of the implementation campaign.

`legacy_oracle.cpp` is the identical test adapter for independently built old/new
AthenaK executables. `run_legacy_equivalence.py` compares every active component
of nonlinear RHS and projection fixtures, and supports a separate one-step RK3
control. Use `--collision OLD_BINARY --current NEW_BINARY --output NEW_DIRECTORY`;
pass `--steps 1` only for the separately authorized numerical control. The
collision control explicitly selects `lapse_projection_target=collision_factorized`;
the current default is `direct_product`. The runner retains exact commands,
input/executable/output hashes and elapsed times. Full CSV data remain outside Git.
