# Staged analytic metric-boundary local qualification

Commit `a15f6f3fe592f1304b6fff9ad707207b3e36226e` evaluates and stores one
symmetric projected metric component at a time.  It retains the exact
projection contraction while removing simultaneous 16-Real metric, 64-Real
metric-derivative, and 50-Real returned-state storage from the PVC kernel.

The unchanged source-unit suite passes.  Closed-loop analytic and generic
one-cycle runs both pass every physical-boundary fence, and their positive-time
histories agree at conditioned Linf `4.86654424958639701e-14` after the same
documented backend-specific Ricci exclusions and paired unavailable-diagnostic
NaNs.

