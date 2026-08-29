# Phase-2 residual-target source-unit checkpoint

This compact local artifact records the first source-unit gate for the direct
physical gauge-target residual.  The production task graph does not dispatch
this path yet.

The gate covered 4,320 generic/analytic backend samples over the full radial
and moving-q matrix.  Exact matched states returned bitwise-zero residuals.
The strict conditioned comparison at radii at least 0.8M passed at
`3.82012e-14` with the fixed `1024 epsilon_binary64` threshold.

`raw-delta-diagnostic=1.3918e-06` is deliberately preserved.  It is the
all-radius difference from the independently evaluated singular subtraction,
which is not a valid near-puncture truth oracle.  The threshold was not
weakened to hide it.  An independent high-precision or generated residual
oracle remains required before production dispatch.
