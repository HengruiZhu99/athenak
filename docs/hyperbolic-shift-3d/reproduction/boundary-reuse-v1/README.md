# V11 boundary execution candidate

Experimental execution optimization, with the same physical PDE, RK stages,
spatial stencils, floor policy and V11 checkpoint schema as the continuous-core
candidate. The default target stays unchanged. The separate target
`hyperbolic_shift_3d_mpi_continuous_positive_fast` enables:

- Reuse of boundary rates for an identical state and timestep. Upload, gathered
  stage/accepted states, and rollback invalidate the cache.
- Replicated small surface projection and angular dissipation on each rank,
  removing projection and angular-residual gathers. Expensive geometric boundary
  and sampled-normal RHS evaluations remain partitioned.
- Separate volume/surface/angular collective timing and call counts.

Steady-state Allgatherv calls decrease from 23 to 12 per RK4 step. This does not
remove volume Allgather or establish a wall-time speedup.

The bounded local audit compares whole checkpoints byte for byte against a
frozen pre-optimization V11 executable. Pulse evolution agrees across one/two
ranks and restart; two-step Minkowski and n16 puncture agree across execution
strategies and partitions. Cache hits occur and angular collective calls are
zero. Reproduce with `tst/hyperbolic_shift_3d/test_boundary_reuse_mpi.py`.

GPU equivalence and timing remain to be measured. No long-time or convergence
admission is implied by these short comparisons.
