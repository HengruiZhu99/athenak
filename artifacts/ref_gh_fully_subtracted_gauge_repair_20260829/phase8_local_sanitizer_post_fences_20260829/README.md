# Phase-8 post-instrumentation local sanitizer discriminator

This compact artifact records the final local Serial discriminator at source
commit `7c575cd690b71783ba0a563768db75e23aa5fd44`.  The build used Debug GCC
with AddressSanitizer, UndefinedBehaviorSanitizer, and Kokkos debug bounds
checking.  Its executable SHA-256 was
`a907d0951a54c41a30a855eed34d541af9e833c77869ca813e03e7fdb3ba8ed2`.

The exact-matched Ref-GH test used STANDARD Phi ordering, gamma0=gamma2=1,
gauge enabled, q=1, FD4, RK4, CFL 0.05, KO 0.02, one 8^3-active-cell
MeshBlock, and one complete RK cycle to t=0.01.  It exited zero without an
ASan, UBSan, or Kokkos-bounds diagnostic.  Every newly instrumented
communication-cleanup entry and exit fence completed during initialization
and after all four RK stages; counts are in `fence_counts.tsv`.

Final stationary-trumpet errors were:

- field Linf: `1.665335e-15`
- physical metric Linf: `4.440892e-15`
- lapse Linf: `7.771561e-16`
- shift Linf: `2.636780e-16`
- constraint Linf: `4.958922e-15`
- RHS estimate: `4.483958e-14`

This is only a reduced local lifecycle/bounds discriminator.  It does not
qualify PVC execution, GPU-aware MPI, the 96^3 case, positive-time trumpet
stability, or convergence.  The complete transient log was intentionally not
committed; `run_tail.txt` preserves the terminal numerical record and
`fence_counts.tsv` preserves the focused lifecycle evidence.
