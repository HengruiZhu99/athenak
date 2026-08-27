# Local staged-q reduction equivalence

Commit `f184fcde3d53b0bed36a855c2a2e07c515abf594` replaced the SYCL-problematic
mixed combined reducer with one current-q sample kernel followed by separate
sum and min/max reductions. The physical estimator, shell, weights, and
controller equations are unchanged.

The retained OpenMP full-output RK4 cycle ran with task fences enabled. Its
trumpet, user, Ref-GH, and six common-ADM history files match the pre-refactor
run exactly (maximum absolute numerical difference zero after normalizing only
the output basename). The accompanying source-unit log also passes the
provider, reprojection, gauge, controller, and reference-cache oracles.

Restart files were intentionally omitted.
