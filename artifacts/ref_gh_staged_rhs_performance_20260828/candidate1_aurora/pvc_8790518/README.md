# Candidate 1 PVC gate, Aurora job 8790518

- Source: `e02e8ced53a66ae45de5615ae8943081c217f8ac`
- Result: `PASS_BOUNDED_EIGHT_TILE_DYNAMIC_Q_CYCLE`
- Node: `x4219c7s2b0n0`
- One/eight conditioned Linf: `1.36191095905485950e-14`
- Tolerance: `5e-12`
- Executable SHA-256:
  `9464a0509b5f4426b988f1d15e5713d960e85d5bef25dcd9e006ed18edea2d6b`

Both one-rank and eight-rank full-output evolved cycles completed with finite
native GH, reduction, and curl diagnostics.  Eight ranks mapped to eight
distinct PVC tiles.  This gate qualifies numerical execution only; the matched
benchmark subsequently rejected the candidate on performance.
