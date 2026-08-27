# KO=0.5 authority and replay record

## Configuration

The three-resolution campaign used native vertex-centered Cartoon SO(2), O4
bulk finite differences, q6 vertex prolongation, RK4, CFL 0.15, KO dissipation
0.50, zero Z4 damping, `dchi_max=0.02`, and derefinement at
`0.25*dchi_max`. The physical outer boundary was 128M. N256 recorded the
accepted hierarchy; N128 and N512 replayed that exact physical-time schedule.

The executable trajectory before CPBC integration was pinned to the recorded
campaign source. The later CPBC discriminator used source commit
`d39822c6522688749fe5ead8025907bc055f02f8` and executable SHA-256
`235aea2e0cb306dc2894d3436e0338f2e9f6e09bdf1e322e29ed296df3afcb79`.

## Accepted hierarchy

- 142 authority snapshots including the initial tree.
- Last event: 141 at coordinate time `14.405106171032207`.
- Final accepted tree: 1052 leaves, logical level 26 (relative level 23),
  checksum `22ff49044e281614`.
- N128 and N512 replayed all 141 noninitial events with `exact_match=true`.

The immutable final authority is
`evidence/authority/n256_ko05_authority_final.jsonl`, SHA-256
`704352de1d33c5e57e7d597589d0956d8b258d66519308646ad8317db02d4f82`.

## Terminal outcomes

| case | scheduler/job | terminal coordinate time | disposition |
|---|---:|---:|---|
| N128 replay | 57634387 | 14.4051061710 | reached forced common tlim |
| N256 record/continuation | 57633404, 57634229 | 14.4051061710 | fail-closed nonpositive metric pivot |
| N512 replay | 57634483 | 14.4051061710 | reached forced common tlim with severe runaway |

N256 failed at cycle 4190, RK stage 1, immediately after the RK update at
`rho=4.8802219629`, `z=0.01611328125`. The conformal metric had a negative
pivot and determinant while `chi=2.135734` remained positive. N512 reached the
forced N256 terminal time but had already developed large constraints and a
collapsed timestep. Reaching that tlim is not a stability result.

Key root-manifest hashes:

- N128: `c83150f16f2f31c70fc66de200d25601a438ea1e3f4b631b99bb02effd5d8a8c`.
- N256 continuation: `5f43b4fae9bc0b3be4f432a4eb32e26e436e0b3a30232fc1a3457b2e6c288b41`.
- N512: `f6566eeeb72eaf28c2f3f329b05724c4eb9b73c415100f7fd1efc1582673fd19`.

No active NERSC jobs remained at handoff.
