# Exact-q=1 boundary dispatch: local validation

This checkpoint specializes the already-selected exact matched q=1 physical
boundary fill at host dispatch. The exact metric kernel performs the same 50
zero writes and four Minkowski-frame diagonal writes as the former runtime
branch. The exact gauge kernel performs the same residual-gauge zero fill. The
general analytic and generic projected boundary paths are unchanged.

A fresh Release/Serial source-unit run passed without weakened tolerances. Key
results remain:

- expanded radial coefficient oracle: 2160 samples, `1.48837e-13`;
- generated geometry oracle: 2376 samples, `2.33147e-15`;
- moving gauge oracle: 2160 samples, `1.24829e-14` motion maximum;
- fully subtracted physical target: 4320 samples, `3.82012e-14`;
- compact boundary oracle: 2160 samples, `4.56474e-14` metric maximum;
- compatible and standard all-61 oracle: 4320 samples, `4.13003e-14`.

A Debug/Serial build with ASan, UBSan, and Kokkos bounds checking completed the
exact stationary-trumpet `16^3` one-step run through all four RK stages to
`t=0.01`. Every stage passed the RHS, RK, prolongation, exact metric boundary,
and exact gauge boundary fences. No sanitizer or bounds diagnostic appeared.
The final field, physical-metric, and constraint Linf values were
`1.665335e-15`, `8.881784e-15`, and `1.010326e-14`.

This is local value-equivalence and lifecycle evidence only. It does not
qualify PVC, positive-time trumpet stability, or convergence.
