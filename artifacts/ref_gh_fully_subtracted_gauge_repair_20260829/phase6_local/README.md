# Phase-6 local residual-production checkpoint

This compact bundle records the first production dispatch of the fully
subtracted gauge equations.  It is a local Kokkos Serial result, not an Aurora
device qualification or an evolution result.

The strict static, uncontrolled, unprescribed `q=1` path now evaluates:

* the physical gauge target as a residual;
* the Hhat/theta driver directly in residual variables;
* `Kref_iA=e_A^a partial_i Href_a` from generated direct expressions;
* the ordinary-GH Einstein gauge increment from `J_a` and `partial J_a`
  residuals, using the same-stage residual Hhat RHS.

All other reference modes retain the legacy production dispatch pending their
own qualification.  The all-61 source-unit oracle nevertheless exercises the
fully subtracted compact evaluator for the complete q/qdot/qddot matrix,
including `dtTheta` and both compatible and STANDARD Phi ordering, against the
independent generic full-reference implementation.

Passing observations:

* deterministic SymPy regeneration was byte-identical in two fresh passes;
* direct generated Kref conditioned error: `1.15471e-15`;
* conditioned residual target/source error: `3.82012e-14` with unchanged
  `1024*epsilon` tolerance;
* all-61 legacy-generic versus residual-compact error: `4.13003e-14` with
  unchanged `256*epsilon` tolerance, 4320 samples;
* exact matched target, driver, gauge source, and gauge KO sectors: bitwise
  zero;
* STANDARD 16^3 production initialization: stored fields/Hhat/theta/Upsilon
  zero and ordinary-gauge Pi increment zero;
* remaining total Pi RHS: `5.681872526233013e-14`, entirely in the existing
  covariant-vacuum source at this resolution.

The all-radius legacy-association diagnostics remain visible:

* physical target: `6.02413e-10`;
* raw target delta: `1.3918e-06`;
* compact/generic residual source: `3.05105e-12`;
* raw reconstructed full driver: `1.05256`.

These near-puncture quantities are not used as unconditioned truth or hidden by
looser tolerances.  The exact matched-zero gate applies at every radius; the
perturbed legacy comparison is gated only in its previously established
conditioned region `r>=0.8M`.

No resolution ladder, evolved cycle, convergence result, or stability result
is contained here.
