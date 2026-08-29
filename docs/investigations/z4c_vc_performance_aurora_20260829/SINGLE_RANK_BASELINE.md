# Single-PVC N512 baseline

## Authority

- Aurora job: `8790557`
- AthenaK source: `f8303c6be7eb214fa1e91b646123ee0d434b3698`
- executable SHA-256: `aae7ccb8739fb4951221ad7be69ea0e220548b52d402086f57d7857fa2c97a13`
- restart SHA-256: `44b8e55957d3b455adf24862d36946e08fc10465df7a30cc5f247ac0e19fa997`
- AMR authority SHA-256: `7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde`
- hardware: one Aurora node, one MPI rank, one PVC tile through
  `gpu_tile_compact.sh`
- workload: the complete frozen N512 hierarchy, 212 MeshBlocks, `64 x 64`
  cells per MeshBlock, starting from the retained cycle-3994 restart near
  `t=9.59463 M`

The run retained the production RK4, CFL, O4 finite differences, P6 native-VC
transfer, KO `0.50`, shock-avoiding lapse, prescribed zero shift, zero Z4c
constraint damping, Sommerfeld RHS closure, replay authority, history cadence,
and binary/restart outputs.

## Result

The bounded run reached `t=9.85 M` cleanly after 88 RK cycles:

| quantity | result |
|---|---:|
| MeshBlock-cycles | 19,928 |
| reported execution time | 46.29792 s |
| zone-cycles/s | 1,763,040 |
| output wall time | 2.623702 s |
| final MeshBlocks | 212 |
| created/deleted blocks | 0 / 0 |

This is the authoritative unoptimized single-tile comparison point. The
existing 24-tile N512 segments delivered a weighted aggregate of approximately
`1.833e6` zone-cycles/s, so the prior 24-tile execution obtained essentially no
aggregate speedup over this one-tile baseline. That comparison diagnoses poor
strong scaling; it does not yet identify the single-rank overhead sources.

Raw evidence is under `evidence/baseline_one_tile/`.

