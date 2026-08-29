# Numerical equivalence

## Final-source single-PVC gate

The required post-change rerun is Aurora job `8790731`, source
`62993e7bac8fbaed13f592834282ca09142a5c2d`, executable SHA-256
`b070bf3b856be712134b0e38028304bbb2fde506aa271350f98b3d8ee243c1e2`.
It reached `t=9.85 M` with 212 MeshBlocks and no topology change.

The independent comparison against unmodified job `8790557` reports:

- complete Z4c history bytes exact, SHA-256
  `de03537e989bfdba1425af13efde52dd060cb5a52fe5d470354df81f962fb0fd`;
- final restart numerical payload exact, SHA-256
  `e498614cad5e50677a1698bc20680e5b34131e0a098c3421e1d64564702c6ab6`;
- identical final time, cycle, MeshBlock count, and maximum level;
- verdict `BITWISE_EVOLVED_STATE`;
- end-to-end speedup `3.379496778x`.

The authoritative offline comparison is
`evidence/optimized_one_tile_v9/evidence/numerical-equivalence.json`, SHA-256
`bd976b3436eeb53ce0d742d1f90aaa0ad200d91491e6356372c050209945b117`.
It was created after the run's immutable root manifest closed and therefore
has its own explicit hash rather than being retroactively inserted into that
manifest.

## Sparse multi-rank gate

Aurora job `8790725` compared the final sparse exchange against the matched
postcondition-only path at 2 and 24 ranks. At both rank counts, the complete
history is byte-identical and the final restart numerical payload is exact.
All four restart payloads have SHA-256
`e498614cad5e50677a1698bc20680e5b34131e0a098c3421e1d64564702c6ab6`.

## Pre-sparse hard-pass gate

The first bounded N512 hard-pass run was Aurora job `8790667`, source
`02a9b4654679bd34d8ae3b06b04245b1da5fba2d`, executable SHA-256
`b4dd1710284c3948a1913e0ff48a3b817d5a4a81ac412ac871de9f97926a7387`.
It reached `t=9.85 M` with 212 MeshBlocks and no topology change.

The independent comparison against unmodified job `8790557` reports:

- complete Z4c history bytes exact, SHA-256
  `de03537e989bfdba1425af13efde52dd060cb5a52fe5d470354df81f962fb0fd`;
- final restart numerical payload exact, SHA-256
  `e498614cad5e50677a1698bc20680e5b34131e0a098c3421e1d64564702c6ab6`;
- identical final time, cycle, MeshBlock count, and maximum level;
- verdict `BITWISE_EVOLVED_STATE`.

The output-staging A/B fixture also produced byte-identical state, ADM,
constraint, and Z4c-diagnostic files.  Its only lean/default directory
difference was the intentionally disabled timestep-contract CSV diagnostic.

Its comparison JSON is
`evidence/optimized_one_tile_v8/evidence/numerical-equivalence.json`.

## Claim boundary

These checks qualify numerical identity over the bounded frozen-hierarchy
benchmark.  They do not establish convergence, Figure-3 reproduction,
long-time stability, or independence from the outer boundary.

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
