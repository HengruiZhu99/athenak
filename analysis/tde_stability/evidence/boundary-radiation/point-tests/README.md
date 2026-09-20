# CPU helper evidence

These standalone C++ point tests were built and run against the v2 boundary helper on local macOS ARM64 with Kokkos OpenMP. The test code accesses Kokkos views directly from the host. It is **not a portable GPU regression**, and it is not an evolution stability test.

`point-build-command.json` records the exact historical local compiler/link command. Rebuilding elsewhere requires a compatible configured AthenaK/Kokkos host build and adapted include/library paths. The unit source includes the current `z4c/z4c_constraint_radiation.hpp`.

Fifteen helper calls passed: equal full/background yields exact zero; poisoned state/RHS ghost cells remain unread; quadratic manufactured residual error1.25e-16; time derivative of the discrete physical Z functional agrees to1.44e-10. Cubic metric convergence exposed v1 first-order behavior and verified v2 second-order convergence. See the original v2 helper hash in `v2-manifest.json`.

These point results do not cure or weaken the negative actual3D evolution result. The v2 experimental boundary develops a rapidly growing face-centered mode.
