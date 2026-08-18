# Remote review prompt

Audit branch `codex/fo-gh-exact-pullback-driver-20260818` independently. Begin
from parent `3a9ba3bb5997e3d3071fed875b2fb0a1672303a8`. Review both the mathematical
argument and the diagnostic code; do not treat the stopped numerical campaign
as a stability result.

1. Re-derive the covector projection and inverse map for
   `h_perp = H_0 - beta^i H_i` and
   `h^i = A chi gtilde^{ij} H_j`.
2. Independently pull back the Lindblom-Szilagyi improved gauge-driver system,
   checking every time-dependent basis/weight term and sign.
3. Test the claimed trumpet scaling with `A ~ r^(2p)`, `chi ~ r^2`,
   `h^i ~ r`, `D0 beta^i ~ r`, and `p=1.091`. Decide whether the normal mixing
   term necessarily scales as `r^(-2p)` and whether any allowed regular field
   definition or exact cancellation invalidates the hard stop.
4. Inspect `exact_driver_pullback_audit.py` for circular or self-confirming
   tests. Compare its component formulas against an independently constructed
   dense weight-matrix oracle.
5. Code-review the optional `fo_gh/fail_closed_dt` telemetry. Check Kokkos
   reductions, rank-local/global interpretation, invalid metric handling,
   state/RHS indexing, host-device synchronization, and default-zero behavior.
6. Verify the compact artifacts and provenance. Confirm that coarse failed at
   3.431611M, medium was explicitly cancelled while finite at 2.108827M, fine
   was never launched, and allocation 57208600 was released.

Report findings by severity with file and line references. Keep observations,
inferences, and hypotheses separate. If the analytic stop is wrong, provide a
concrete corrected regular variable map and finite exact RHS before proposing
new runs. If it is right, recommend the smallest formulation-level redesign;
do not tune damping, dissipation, floors, masks, or boundaries around it.
