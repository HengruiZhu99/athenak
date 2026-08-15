# Figure-3 enlarged-domain KO/shift controls

This is a prospective two-case successor to Perlmutter job `56993758`. It
reruns that campaign's later two controls only after the complete three-case
result has been authenticated. The predecessor's dependent four-GPU v1
allocation (`56994026`) was cancelled before allocation when its attached SSH
submission timed out; no science ran. The unsubmitted v2 regular-QOS design is
superseded by the user-directed one-GPU design. Its first v3 staging attempt
then stopped before promotion because the cancelled-v1 ledger named
`sacct-current.psv` after the immutable finalizer had renamed it to
`sacct-settled.psv`; v3 remains preserved as an unlaunched `.incomplete` root.
This fresh v4 changes only that provenance filename and identity.

V4 passed login preflight and was never launched.  Before launch, the
predecessor exposed a second operational issue: its compute-node finalizer
spent many minutes hashing 9,264 raw slice files while retaining the GPU
allocation.  This fresh v5 keeps every science/input byte unchanged, but the
compute wrapper now records only exit/source status before releasing the A100.
The outer allocation wrapper then records settled accounting and hashes the
complete evidence tree after `salloc` returns, when no GPU is held.

V5 passed login preflight but its launch stopped before `salloc`: it began
checking the predecessor just as the predecessor's outer wrapper replaced the
earlier in-allocation manifest with the final accounting-inclusive manifest.
The verifier therefore saw a manifest being rewritten and failed closed.  V5
has no run tree or job.  This fresh v6 preserves that failure and does not wait
for the redundant 131-GiB raw-output rehash.  Instead it freezes and verifies
the terminal compact record directly: settled accounting, comparison JSON,
all three histories/logs/commands/results/exits, twelve rank bindings, run
evidence, source status, and the original bundle/preflight identities.  Raw
slice and restart bytes are not prerequisites to launch the independent
enlarged-domain cases and remain preserved in the predecessor root.

V6 then exercised the requested one-GPU design on an A100-SXM4-40GB.  Both
cases imported the same IrisK data and reported the same finite initial
constraint diagnostics, but each aborted before evolution when Kokkos could
not allocate the 4.883-GiB load-balance send buffer.  The 16,384-block
single-rank capacity preallocates both send and receive buffers; together with
the resident Z4c state this exceeds 40 GiB.  Reducing capacity would recreate
the predecessor's known block-capacity stop.  This fresh v7 therefore retains
one rank, one GPU, the shared-interactive QOS, and every science byte, changing
only the node-memory constraint from `hbm40g` to `hbm80g`.

The compute wrapper requires the predecessor's frozen selected-evidence ledger
to self-verify, exactly three terminal numbered steps, all three histories/logs,
and the strict comparison schema before either enlarged-domain case can start.
The two cases then run strictly serially in one
`shared_interactive` allocation using one rank and one A100; they never overlap.

The only common grid-domain change is

```text
rho: [0,16] with nx1=64   -> [0,64] with nx1=256
z:   [-16,16] with nx2=128 -> [-64,64] with nx2=512
```

The base spacing remains exactly `0.25` in both represented directions, so the
test changes the outer-boundary distance without coarsening the central grid.
MeshBlocks remain 32x32, giving 128 rather than 8 root blocks.  The unchanged
AMR criterion remains `dchi_max=0.02` with physical levels 0--20.  Because
the KO=0.5 predecessor reached 8,126 blocks and the enlarged root adds 120
coarse blocks, the one-rank capacity and global guards are both 16,384 blocks.
This is deliberately large enough to test the user's one-GPU hypothesis
without turning rank capacity into an earlier bookkeeping stop; it does not
alter refinement selection or resolution. GPU memory exhaustion remains a
valid one-GPU scaling result and will not trigger an adaptive retry.

Both cases retain the exact source/executable, IrisK `A=-0.047` data, N128/O6/
RK4 setup, pre-collapsed lapse, telegrapher `(tau,kappa)=(1,1)` with max-|K|
scaling, KO `diss=0.5`, target `t=20`, one rank/one A100 GPU, and zero Z4c
constraint damping. They run sequentially on one A100-SXM4-80GB:

1. fixed Gamma-driver shift with `eta=2`;
2. zero shift.

No source/build change, floor, threshold relaxation, or adaptive retry is
authorized.  Failures remain evidence.  The eventual comparison must treat the
original-domain and enlarged-domain cases separately and may only attribute a
difference to boundary distance after confirming the same base spacing and all
other parameters.
