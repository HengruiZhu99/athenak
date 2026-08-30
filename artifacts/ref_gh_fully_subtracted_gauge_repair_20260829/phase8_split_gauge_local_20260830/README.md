# Phase-8 split-gauge local qualification

This checkpoint tests the equation-preserving portability correction that runs
the analytic gauge-driver RHS in the existing separate active-cell kernel
instead of duplicating it inside the scalar-source/Pi kernel. No continuum
term, finite-difference stencil, RK stage, parameter, persistent reference
allocation, or diagnostic definition changes.

The fresh Release/Serial source-unit run passed every existing oracle without
a tolerance change. In particular, all 4320 compatible and STANDARD all-61
comparisons retained conditioned maximum `4.13003e-14`. The coefficient,
expanded radial, generated geometry, moving mixed-jet/`dtTheta`, residual
physical target, and compact boundary gates also retained their established
values.

A Debug/Serial build with ASan, UBSan, and Kokkos debug bounds checking then
completed the exact-matched STANDARD, gamma0=gamma2=1, gauge-enabled q=1 case
for one complete RK4 cycle on a 16^3 grid through `t=0.01`. Every one of the
four new separate gauge-driver stage fences, all main-RHS fences, RK updates,
and communication cleanup entry/exit fences completed. No sanitizer or bounds
diagnostic was emitted.

Final errors were:

- field Linf: `1.665335e-15`
- physical metric Linf: `8.881784e-15`
- lapse Linf: `7.771561e-16`
- shift Linf: `3.469447e-16`
- constraint Linf: `1.010326e-14`
- RHS estimate: `5.681873e-14`

This qualifies the algebra and reduced local lifecycle only. It does not prove
the PVC fault fixed and supplies no trumpet-stability or convergence evidence.
