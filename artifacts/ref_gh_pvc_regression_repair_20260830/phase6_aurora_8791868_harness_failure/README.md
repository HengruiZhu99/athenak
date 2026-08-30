# Aurora job 8791868: setup-only failure

This job exited in one second before configure/build because the archived
Git submodule placeholder caused the pinned Kokkos symlink to appear as
`kokkos/kokkos`. The PBS output records the exact failed preflight check.

No executable was built and no numerical workload ran. This job has no PVC
scientific or portability classification. The malformed directory was moved
aside, not deleted, before the exact same source markers were retried as job
`8791879` with the correct pinned Kokkos path.
