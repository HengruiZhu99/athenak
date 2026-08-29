# Staged all-61 oracle local regression

This compact checkpoint verifies the equation-preserving portability response
to Aurora job `8791211`'s IGC failure.  The independent legacy-generic and
fully-subtracted compact 61-component RHS evaluations now execute in separate
device-dispatch kernels, followed by the unchanged conditioned comparison.

The local Kokkos Serial build and complete source-unit run passed.  The all-61
result remains `4.13003e-14` over 4320 samples with compatible and STANDARD
Phi ordering.  The tolerance remains `256*epsilon`; no samples, equations, or
conditioning rules changed.

This is host regression evidence only.  It does not establish that the staged
kernels compile or execute on Aurora PVC; that remains the focused rerun gate.
