# Phase-7 residual-target diagnostic local checkpoint

This checkpoint adds a diagnostic-only evaluation of the already-qualified
cancellation-free physical gauge target on the actual stored stationary state.
It records `deltaF_A`, delta conformal Gamma, and delta shift separately from
the retained legacy full-target cancellation diagnostic.

A fresh local Serial rebuild and 16^3 exact matched STANDARD fixed point pass.
The new residual audit reports exact zeros for `deltaF_A`, delta conformal
Gamma, and delta shift. Stored Hhat/theta, actual and driver gauge RHS sectors,
ordinary-gauge Pi increment, and all gauge KO sectors are also exact zero. The
remaining total Pi RHS is `5.68187252623301281e-14` and is entirely the
existing covariant-vacuum source.

The unchanged source-unit suite passes, including all 4320 all-61 comparisons
at `4.13003e-14`. The new ladder analyzer was syntax-checked with local Python
3.12.3 and Aurora Python 3.6.15 and smoke-tested using the local exact-zero
record. No positive-time evolution was performed.
