# Aurora Phase-6 rank-contract gate

Job `8791879`, node `x4020c0s7b0n0`, project `CompactBinaryMerger`, debug
queue. Exact source commit `d5e671453c3eb45e96ab9455ee06714694d774b3`,
tree `39f4dc8981577499b9dce69636fade06cd44fd85`, Kokkos
`6739bc623081648af9e752b616d9671527922cbf`.

The source includes the generic uniform-grid rank-packed message-size
correction but retains the default production SYCL device-code-split policy.
One build and executable were used for both rank counts.

| Ranks | Blocks/rank | Classification | Latest time |
| --- | --- | --- | --- |
| 1 | 216 | `PASS_ONE_CYCLE_PVC` | `0.003404497M` |
| 12 | 18 | `FAIL_LEVEL_ZERO` | `0M` |

The one-rank log completes every evolved fence and final diagnostic. The
twelve-rank log contains two Level Zero PDE `NotPresent` writes after all
ranks had reported the projected trumpet metric and gauge boundary fences.
This demonstrates that the contract fix is not sufficient to restore the
twelve-rank production image. It is not a scientific stability result.

Executable SHA-256:
`58964e2b6843c0051726e7accda6235bd159d5a84ab471996f7ebc5e344bd614`.
Input SHA-256:
`6d483ded11b70d640f4a166fd21757f956802622d4e3994ceda71ae8649235eb`.

The harness stopped on the nonzero twelve-rank launcher status before writing
its normal `result_n12.txt`; the committed compact result is reconstructed
only from the preserved raw log/PBS status and labels that fact explicitly.
