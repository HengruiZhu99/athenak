# V11 puncture pilot — completed evolution, explicit metadata correction

Native evolution completed coordinate t=0.15 in 300 steps, dt=0.0005, n16,
159200 evolved cells, source-v19/faee66ea3, one debug node with 12 ranks/tiles.
Wall time 222.703 s; no lapse-floor interventions.

The original worker flag and independent verification.json are **failed** solely
because the copied plan required segment_steps=2001 and cells=189512. The plan
also already required target t=.15, steps=300, V11 geometry, and cells=159200.
Those contradictory inherited counts cannot both hold. All numerical criteria
passed. They were not changed. pilot-metadata-audit.json separately qualifies the
bounded pilot using counts derived from the prescribed time step and fixed V11
mesh. It preserves the original failure, checkpoint SHA, source/binary identity,
and exact correction. The native checkpoint is admitted and finite.

Replay the correction with ../../continuous-positive-v1/audit_pilot_metadata.py.
The independent full artifact/checkpoint audit is in verification.json; the
underlying reader is ../../evolved-overlap-v1/verify_duration.py with the pinned
../preflight-continuous-positive-v2/verification/verify.py decoder and admission.
This is not a 3M, trumpet, convergence, AMR or nonspherical stability result.
