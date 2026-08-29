# Numerical equivalence

## Matched N512 comparison

The authoritative comparison uses the same retained N512 restart, complete
212-MeshBlock frozen hierarchy, physical interval, timestep policy, output
cadence, and evolution configuration. The baseline is Aurora job `8790557`;
candidate 1 is job `8790595`.

The repository restart contract compares bytes after the embedded
`<par_end>` parameter terminator. This excludes the intentionally different
observational/runtime settings while retaining the complete serialized
evolved state and MeshBlock payload.

| contract | result |
|---|---|
| complete Z4c history bytes | exact |
| final restart numerical payload | exact |
| final coordinate time | exact, `9.85 M` |
| final cycle | exact, 4088 |
| final MeshBlocks | exact, 212 |
| final maximum physical level | exact, 5 |
| accepted hierarchy changes | exact, none |

Both the history and restart payload have the same hashes in the two runs:

- history SHA-256:
  `de03537e989bfdba1425af13efde52dd060cb5a52fe5d470354df81f962fb0fd`
- restart numerical-payload SHA-256:
  `e498614cad5e50677a1698bc20680e5b34131e0a098c3421e1d64564702c6ab6`

The machine-readable comparison is
`evidence/optimized_one_tile_v3/evidence/numerical-equivalence.json` with
SHA-256
`a617c17e93235d23cc3d284e8c83cd4b8eb4b5983fe17861180fc3c8bd8dbdd0`.

## Default-off compatibility

A separate local run compared pre-optimization commit `d35c8248` with the
candidate while omitting the lean selector entirely. Its complete Z4c history
and final restart numerical payload were byte identical:

- history SHA-256:
  `c19405bf9c7128be4da403d62fbe6664de3432691ef42abd4ad559c053c44e93`
- restart numerical-payload SHA-256:
  `7e1287a34d0db96857cdfc5e650774066f0b8feeb2ac2a303fd39dcfb4a1a107`

This is a numerical-equivalence result only. It is not a convergence,
Figure-3 reproduction, stability, or physical-criticality claim.
