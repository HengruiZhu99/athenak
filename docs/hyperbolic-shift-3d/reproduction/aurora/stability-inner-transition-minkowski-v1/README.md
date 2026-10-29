# V7 Minkowski coordinate-time2 stability control submitted

Fresh job **8984359** uses the GPU-qualified V7 binary on one
CompactBinaryMerger/debug node, twelve MPI ranks and twelve GPU tiles.
The initial state is analytic Minkowski, n16, dt=.0005, epsilon2, target
t=2. It starts fresh and does not resume any failed or bounded checkpoint.

The state/H/M/null/positivity/algebraic/CFL limits are unchanged from the V6
duration protocol. The longer interval tests the candidate beyond V6's t=1
result. The native plan additionally requires V7's explicit inner-transition
interpolation identity. **No duration result or stability claim is available
yet.** Source-v14, the exact binary, plan and independently verified preflight
admission are hash-bound in the retained files.

Submission follows independent verification and release of
[preflight8984332](../preflight-inner-transition-v1/README.md). No second
allocation was submitted concurrently. Long-time puncture stability,
100 M reproduction and convergence remain unproven.
