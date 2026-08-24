# Split-rank native-VC derefinement

## Red result

The bounded Perlmutter discriminator used three 2D sibling families and an
equal-cost partition. The middle family has old children 18:21 and new parent
15. With two ranks, the lower child and target parent are on rank 0 while the
upper siblings cross to rank 1. This is the required layout B.

On commit `941a386f090ff6f04910198ea24bb3f9ee2f12af`, the one- and two-rank
executions completed the same physical hierarchy and event times but their
post-derefinement fields differed at 3,173 binary-output values. The maximum
absolute difference was `2.3450449225492775e-04` and the RMS difference was
`6.495238513420746e-06`. Parent GID 15 contained 2,565 of the differing
values; four neighboring blocks differed after boundary reconstruction.

The first recorded mismatch was `chi` at parent-adjacent stored index
`[gid=8,k=0,j=16,i=0]`: one-rank `1.0000348091125488`, two-rank
`1.0000083446502686`. The rank-canonical payload hashes were:

```text
one rank  0fb5b6c8eb5402a3d8b0ef9a179e89f061e573ce13612541651f0be72a83607c
two ranks 7ababd1b72a684a530bf3a7e441ae7da32eb992c894f72107462dff238cd3c50
```

Perlmutter job `57499665` completed in eight seconds. The CPU/OpenMP MPI
executable SHA-256 was
`e2586a6b04d5db21050ac3f5b6f9e3134561deaa9bccd92142631ad87f45c3dc`.

Observation: the target parent is rank dependent. Source audit shows that
`DerefineVCSameRank` skips the complete family when any sibling is remote,
while `InitRecvAMR` receives only the remote upper siblings when the lower
child and parent remain local. A6 therefore copies the lower child's full
active array into the parent, and A8 overwrites only the received quadrants.

## Green result

Pending deterministic split-parent assembly and the 2-/4-rank matrix.
