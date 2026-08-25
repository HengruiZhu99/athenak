# Rout=128 authority and replay qualification

## Input and executable authority

The large-domain experiment used `brill_vc_rout128_smr.athinput`
(SHA-256 `1aa50746d12e51c91c3e7adbc4728e325a73ce98e687c6aaf218df70b4eb4699`)
with the existing Perlmutter executable
`3a395bfdaf217d617fee43d2cbcd38e7a13c2a0f4207e3a764c3513eb8c0405f`.
The executable source is
`d63519328214a6315a9cc1f7d5e4a1aa4bca21b0`, tree
`9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4`. The source worktree was
clean. No production numerical source was changed for this campaign.

The input preserves diss=0.02, zero Z4 damping, O4 spatial differences, q6
vertex prolongation, RK4, CFL 0.15, telegrapher lapse, Gamma-driver shift,
dchi_max=0.02, and derefinement factor 0.25. Only the domain, persistent SMR
minimum levels, and absolute level ceilings differ from Rout=16.

## N256 record authority

The N256 record run completed at exactly `t=6.5M` with exit 0 in allocation
57566483 on one NVIDIA A100-PCIE-40GB. It used 1 MPI rank, 1 GPU, 32 bound CPU
cores, and reported a 5,815 MiB monitored HBM peak.

The accepted authority contains:

| event | time | cycle | leaves | logical max level | tree checksum |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 104 | 6 | 6d321941cbd19639 |
| 1 | 0.0108233129374 | 1 | 200 | 7 | 24316947a3a67cd8 |
| 2 | 0.0162141754365 | 2 | 212 | 8 | cf0d2384b11c1d42 |
| 3 | 0.297991949657 | 111 | 206 | 7 | 0ca3eb7a28f554f5 |

The maximum physical refinement level reported in history is 5 and the
maximum leaf count is 212. The calculation ends with 206 leaves, 108 created
and 6 deleted. There is no refinement explosion, inadmissible state, transfer
failure, or SMR bookkeeping failure.

## Same-resolution identity replay

A separate N256 replay in allocation 57568295 reached `t=6.5M` with exit 0.
Every replay event reported `exact_match=true`, including the final
206-leaf checksum. The record and replay 264 by 71 history arrays are bitwise
identical:

`701f3093893017ea064087e96ddc0dbcd4136cae36b50f0330c657d98c5fb9c5`.

The maximum absolute and relative history differences are both zero. This is
the required deterministic record/replay identity gate.

## Cross-resolution replay

N128 and N512 replayed the N256 authority at the recorded physical times.
Both ended at `t=6.5M`, exit 0, with the exact requested 206-leaf hierarchy.
N128 used 5,815 MiB monitored HBM. N512 used 36,581 MiB, fitting on one 40GB
A100 without escalation. The replay shadows disagree with the N256 native
criterion, as expected for different cell resolutions; the accepted tree,
not the shadow flags, is the convergence authority.

All raw histories, authority files, replay ledgers, bindings, scheduler
metadata, commands, and logs retained for review are under
`evidence/perlmutter/runs/`. Multi-gigabyte state/restart files and verbose
native-shadow streams remain at the manifest's Perlmutter paths and are not
committed.

