# Self-contained local CTest contracts

Root CMake registers `elliptic_shift_3d_enriched_log_value` and
`elliptic_shift_3d_affine_shift_remainder` in the existing optional elliptic-shift
component build, with 30-second test timeouts. Only these two targets are built;
only these two tests are selected. Both build and CTest use at most two jobs.
No external initial.dat, persistent geometry cache, production evolution, GPU
allocation or full-grid solve is required.

The value contract checks independent off-grid polynomial-plus-log manufactures
at three puncture positions and two spacings, including mixed Cartesian and
fifth-degree terms. It also checks exact active identity and constants, input
preservation, stale point/kinematics/stage/revisions, NaN/shape rejection without
output mutation, invalid geometry, rank deficiency and value-weight amplification.
The affine contract is the previously isolated 256-case representation/arithmetic
falsifier; registering it does not integrate an augmented elliptic solve.

To reproduce when the shared CPU build directory is idle:

    python3 docs/elliptic-shift-3d/evidence/enriched-log-value-contract-20261027/reproduce.py

The recorded run uses `--test-only` because configuration and the two-target
`-j2` build had already succeeded directly in the same delegated turn. CTest
execution was deferred while another agent owned the native production rebuild.
The manifest records those observed prior commands and the actual CTest command.
`ctest.xml` retains test status and CTest-limited stdout. Because JUnit truncates
the large passing affine output, `full-ctest-last-test.log` freezes the complete
`build-shift-3d/Testing/Temporary/LastTest.log` from this run. The compact
`contracts-summary.json` combines the value summary with the affine summary
extracted from that complete log, without rerunning either binary. These are local safety
and correctness contracts, not new coupled scientific acceptance results.
