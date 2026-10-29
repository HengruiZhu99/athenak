# Joint observed-direction control

A read-only four-dimensional basis combines the observed V6 and V7 endpoint
directions with their own native tangent RHS directions. Both endpoints and
the fixed arithmetic midpoint of their tangent operators are compared on this
same basis. The weight1/2 was fixed before inspecting the result; no weight
scan or evolution change was made.

The midpoint has projected pairs −.34464±565.05977i and−6.41701±572.11001i.
Their full-action relative residuals are7.05% and10.22%, respectively. These
residuals are too large to regard the negative real parts as verified damping
of full eigenmodes. This is not a stable interpolant or a recommendation to
adopt averaging. A larger subspace/action check is needed before drawing a
numerical conclusion. V6 and V7 retain their known positive projected pairs
on this joint basis, approximately15.50±554.21i and10.816±603.92i.

Perturbation amplitudes1e-5 and1e-6 give consistent results. The initial BLAS
matrix products emitted floating-point warnings despite finite output; those
raw records are preserved in v1. A complete fresh action/reduction replay in
v2 uses explicit scalar reductions instead. It emits no warnings and agrees
with every projected matrix to the tolerance recorded in
`reduction-recheck.json`. Use v2 as the verified arithmetic result. Each run
completed in about62s under a385s external bound, with one CPU process.

The library's45 captured sources match frozen V7 exactly. Both checkpoints
are independently decoded and worker-bound before use; neither is evolved or
restarted. The scripts, imported-source hashes and execution records are
retained. From the repository root, with the recorded local payloads present:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/stability-inner-transition-minkowski-v1/interface-span/reproduce.py --output output/interface-span-recheck-new
```

The wrapper refuses an existing output directory and imposes an external385s
process timeout in addition to the native-call deadline.
