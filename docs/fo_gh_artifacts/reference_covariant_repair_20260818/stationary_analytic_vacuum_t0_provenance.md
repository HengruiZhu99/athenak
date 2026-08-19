# Post-curvature stationary t=0 executable ladder

Source commit: `9c2e3579953b796b27c62bc6c6a9b07b3a6711bb`, a clean detached
worktree containing the analytic stationary Schwarzschild frame-Riemann
provider and source-sector history diagnostics.  The documentation-only commit
`0c7f9b40` was not part of either executable.

The serial executable SHA-256 is
`a695441eea276d4f534328e651b6405a1834d0382006d3d111f4203409f680ba`.
It used Release, Kokkos Serial, and no MPI.  The OpenMP executable SHA-256 is
`01e81c145f732ff21a0a08694fd7106dfa58902366aca1ed20e1d67908dd7db9`.
It used Release, Kokkos OpenMP, no MPI, and `OMP_NUM_THREADS=32`.

The three invocations differ only in resolution, MeshBlock dimensions, and the
documented execution backend:

```text
athena -d <fresh> -i inputs/ref_gh/ref_gh_stationary_trumpet.athinput \
  time/tlim=0.0 job/basename=t0_n{64,96,128} \
  mesh/nx{1,2,3}={64,96,128} meshblock/nx{1,2,3}={32,48,64}
```

For the final `128^3` line only, the OpenMP executable was launched with
`OMP_NUM_THREADS=32 OMP_PROC_BIND=false`.  The complete terminal records are
the adjacent `stationary_analytic_vacuum_n*_t0.txt` files.  `tlim=0` evaluates
the full lower-order stationary RHS and performs the final field/native-
constraint diagnostic refresh without advancing a time step.
