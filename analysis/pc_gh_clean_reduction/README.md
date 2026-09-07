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
