# Static refined-core component checks

The V7 native device path also passes the existing complete halo/RHS and
scri checks on two statically refined Cartesian meshes: eight children of
the original core (26 total blocks), and one of those children refined again
(33 total blocks, with a Cartesian coarse/fine interface). All 18 shell
blocks retain their geometry and resolution. These tests use n6 per block.

The tests cover Minkowski and a smooth nonspherical perturbation. CPU/native
Serial device differences are zero for the full halo, bulk RHS, sampled
numerator and boundary rate. Null residuals remain below1.12e-15. Twelve
simulated boundary ownership partitions agree exactly; exterior ghost poison,
dependent-D invariance, and invalid-input rejection checks also pass. The
unchanged unrefined test passes after removal of hard-coded shell block IDs.

This is a component check on the local Serial backend, not actual GPU or
MPI execution, time evolution, interpolation convergence, dynamic AMR, or
checkpoint support on a refined mesh. It removes one obstacle to a separate
core-resolution experiment without claiming that experiment is implemented.
The currently running Aurora V7 source-v14 binary is unchanged by these tests.

Build and run from the repository root:

```sh
cmake --build build-hyperbolic-mpi --target hyperbolic_shift_3d_refined_core_device -j2
ctest --test-dir build-hyperbolic-mpi -R '^hyperbolic_shift_3d_refined_core_device$' --output-on-failure
```

The CTest timeout is420s. Exact executable hashes, runtimes and logs are
retained here. The expanded test uses the same CPU equations as its oracle;
execution parity alone cannot establish the correctness of those equations.
