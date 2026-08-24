# Repaired native-VC Brill authority qualification

## Verdict

`NATIVE_AMR_UNSTABLE`

The derefinement repair removes the immediate post-event-3 oscillation and
supports clean N128/N256/N512 common-tree convergence through central proper
time tau approximately 4. It does not produce a healthy native authority to
tau approximately 7.

## What passed

- Source authority and numerical diff were verified before execution.
- Post-repair host suite: 140/141 passed concurrently; the sole fixed-timeout
  AMR-history test passed in isolation in 177.96 seconds. Two CUDA-only tests
  were disabled.
- Focused one-A100 suite after the diagnostics repair: 20/20 passed.
- N256 record/replay: exact hierarchy, histories, auxiliary diagnostics, and
  post-header numerical binary/restart payloads.
- Repaired authority: exact agreement with the historical authority through
  event 3, then the expected absence of the erroneous immediate re-refinement.
- N128/N256/N512 through tau approximately 4: monotone C/H/M/Z improvement,
  positive effective order, convergent nonzero evolved fields, positive chi,
  positive conformal-metric pivots, roundoff/zero shared-node discrepancies,
  and protected causal comparison regions.

## What failed

On the N256 continuation, C begins sustained growth on the unchanged 44-leaf
hierarchy at tau about 4.45 and exceeds 1 by tau about 5.61. The first late
refinement is accepted only at tau about 6.28. A subsequent refinement cascade
reaches logical level 23, physical level 20, 1,367 leaves, and
`dt=1.07e-8 M` at `t=11.192887945 M`, while C reaches 1.40e14 and domain max
|Kretschmann| reaches 1.60e25. Job 57525753 was cancelled at that fail gate.

The observations strongly show that the instability is already developing
before the refinement cascade. They do not yet decide whether the primary
cause is parent under-resolution/poor chi sensing, a continuum or bulk
discretization mode, or later transfer/interface feedback. The next decisive
diagnostic should use the preserved pre-event-4 state to compare all evolved
field resolution sensors and transfer provenance; broad tuning is not
justified by this campaign.

## Qualification boundary

No convergence claim extends beyond tau approximately 4. No tau 10.5 or full
published-interval run was attempted. This is not a Figure-3 reproduction and
not evidence of critical behavior. The failed authority is preserved for
diagnosis only.

See `analysis/native_summary.json` for machine-readable results and
`EVIDENCE_MANIFEST.json` for exact paths and hashes.
