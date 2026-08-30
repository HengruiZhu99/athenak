# Target-scoped SYCL per-kernel qualification

Exact source: `8644c97579ff9725aea50e7abcde10b729c9697c`.

Correction: apply `-fsycl-device-code-split=per_kernel` to compilation of the
`athena` executable target and to its final link. Kokkos support-library
targets retain their normal policy. No source arithmetic, equations, task
ordering, boundary values, or numerical parameters changed.

| Job | Queue | Node | Classification | Time | Executable SHA-256 |
| --- | --- | --- | --- | --- | --- |
| 8791922 | debug | x4217c0s5b0n0 | PASS_ONE_CYCLE_PVC | 0.003404497M | e01b3239d63345cc47bbf8b4986e54c551eab772f3e04f6c4d35b2561aace5c1 |
| 8791932 | debug | x4217c0s5b0n0 | PASS_ONE_CYCLE_PVC | 0.003404497M | a2c75bb689bb560bce9a47f4e15fcf9898a9864906a30368052c2174e2a7b8b0 |
| 8791941 | debug-scaling | x4217c0s5b0n0 | PASS_ONE_CYCLE_PVC | 0.003404497M | ed508b52bf344c9304d0655dc09d51fc1abbb4aeee1562d8b1e6492be26fbbcf |

All jobs used twelve distinct PVC tiles, completed all twelve final history
fences, and had zero Level Zero error messages. Despite independently built
executables, the three Ref-GH histories are bitwise identical with SHA-256
`55f63d81fb3eb56ff8a5333e4cb4f342f139fd39abe3b913180de8d5354fd9ca`.
The common-ADM0 histories are also identical with SHA-256
`6c50a7746dc6c8639d8896657cbfdb68b302921f23ca6c485bbec276e0e51a35`.

Pass 1 compile provenance: 189 of 214 compile entries carried the flag,
including nine Ref-GH sources; no Kokkos source carried it. The final link
carried the flag once. Source-only job 8791895 and link-only job 8791910 both
failed, so compile-plus-link target scope is the smallest tested passing
configuration.

Claim boundary: `PVC ONE-CYCLE PORTABILITY RESTORED`. All three fresh jobs
landed on the same node, so there is no multi-node sample. This evidence does
not establish evolution stability or convergence.
