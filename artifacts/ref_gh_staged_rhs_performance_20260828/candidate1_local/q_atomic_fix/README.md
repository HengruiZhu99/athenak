# Compact atomic q accumulation local qualification

Commit `ddd0d780513c4a2b3cbef3df2f990694d29bb3ce` replaces the PVC-failing
custom 32-Real Kokkos reducer with one device kernel atomically accumulating
the nine active estimator moments into a compact view.  The estimator formulas,
eligible cells, weights, controller, and single MPI collective are unchanged.

The full source-unit suite and a closed-loop evolved cycle pass locally.  On
the Serial backend, every retained history value is identical to the preceding
custom-reducer run (conditioned Linf zero, treating paired diagnostic NaNs as
equal).  Aurora PVC remains the hard portability gate.

