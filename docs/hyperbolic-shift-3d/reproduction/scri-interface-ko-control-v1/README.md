# Fixed doubled-bulk-dissipation control does not qualify as a repair

A read-only native adapter compares the existing conservative bulk KO8
coefficient.5 with the single fixed doubled value1. The PDE, halo policy,
scri closure and all evolved variables are otherwise identical to V7. No
production evolution executable or checkpoint schema changes.

On the same four-vector span of the observed V6/V7 disturbances, the leading
V7 pair changes from10.81596±603.91754i to **2.61653±603.13877i**. Its
relative full-action residual at KO1 is1.60%, with absolute residual9.65181.
The other pair is−26.33004±574.12959i, residual11.54%. Two perturbation
amplitudes agree. Positive projected growth remains, and the residuals do
not establish full eigenmodes. This control does not justify adopting doubled
KO as a stable scheme or launching a GPU duration test of it.

The prototype library differs from the retained frozen CPU library only in
the diagnostic adapter. Five complete RHS arrays, including a coefficient
reset, match the old V7 library bitwise at coefficient.5. Invalid coefficient,
nonfinite value and null-handle tests are rejected. The baseline projected
matrix replays the previous V7 result. Runtime is82.05s under a385s bound.

`source.json` and the captured adapter identify the exact prototype delta;
all remaining source snapshots match the retained library. The build command,
input/import hashes, full results and execution record are retained. The
local library remains under
`output/hyperbolic-core-shell/scri-v7-ko-control-build-v1`; it is a diagnostic
library and is never used to restart or evolve a saved checkpoint.
