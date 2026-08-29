# Phase-5 exact matched-state initialization

The exact stationary q=1 predicate and initialization paths are tested here.
The source-unit suite verifies that the predicate is disabled by every moving,
controlled, prescribed, or nonidentical case.  The focused initialized mesh
uses the analytic radial-q backend with a 16^3 single MeshBlock.

Observed exact stored-state results:

* field Linf: `0`;
* stored Hhat residual Linf: `0`;
* stored theta residual Linf: `0`;
* physical metric Linf: `2.22045e-16`;
* physical lapse Linf: `1.11022e-16`;
* physical shift Linf: `3.33067e-16`.

The initial RHS remains `5.80111e-14` because production still uses the legacy
full gauge driver/source reconstruction at this checkpoint.  It is not a
fixed-point qualification and no evolution claim follows.
