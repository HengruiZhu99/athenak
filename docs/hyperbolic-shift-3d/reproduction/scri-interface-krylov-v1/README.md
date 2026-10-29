# The four-vector midpoint result does not survive a larger subspace

The fixed arithmetic mean of V6 and V7 tangent operators was examined in a
32-vector block-seeded Krylov subspace. No interpolation or evolution was
changed. The initial four-vector projected matrix reproduces the previous
joint-plane control exactly, and basis orthogonality error is1.55e-15.

| Subspace dimension | Largest projected real part | Relative full-action residual |
|---|---:|---:|
|4|−.34464|7.05%|
|8|15.4560|4.49%|
|12|205.9551|22.83%|
|20|150.2505|15.86%|
|32|80.3076|3.73%|

These values are **not converged eigenvalues**. At dimension32 the leading
pair is80.30763±651.37588i, with absolute residual24.51958. Independent full
actions at perturbation amplitudes1e-5 and1e-6 reproduce that residual. The
small four-vector projection's apparent decay cannot support adoption of an
averaged interpolant. No GPU evolution of that candidate is scheduled.

The calculation uses the retained native CPU library, with all45 source blobs
matched to frozen V7, and independently decoded worker-bound V6/V7 snapshots.
It performs74 evaluations of each operator and completes in213.19s under a
385s external bound. The result includes all projected matrices, residuals,
source/input hashes and the full command. It does not claim that this Krylov
sample covers the complete spectrum or proves instability of a nonlinear
interpolation scheme that has never been implemented.

The script uses explicit scalar reductions, avoiding the BLAS warnings found
in the earlier four-vector control. To repeat, run the recorded command from
the repository root with the retained local libraries/checkpoints, a fresh
output file, and the same385s external process bound.
