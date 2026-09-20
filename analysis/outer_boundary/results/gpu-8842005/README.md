# Repaired-stencil GPU verification: PASS

Job8842005 completed with PBS exit0 in1m03s, using immutable repaired GPU executable SHA256 `a6c3af79571819fba5dc2252ceb9abacb31279feec440f2c542e342dfee43639`.

- Exact-zero residual vacuum stayed exactly zero through3RK3cycles on1/2/4MPI ranks, including final ghost-inclusive checkpoints. Every payload was finite and the full spatial metric remained SPD.
- The matched finite oblique lapse pulse triggers144 physical-face/internal-tangential-edge samples per rank layout. Raised tangential normals span9.46381e−6..0.00409138; the boundary changes the RHS by0.016271. All ten boundary-rate equations pass independently, maximum error1.73472e−17.
- All signed active stage arrays and full ghost-inclusive postRK residual blocks are bitwise identical across GPU MPI1/2/4 layouts.
- CPU/GPU comparison passes all360 arrays with per-element tolerance abs(error)≤1e−12+1e−10abs(CPU). Maximum absolute error is2.38628e−13, and maximum error/tolerance0.215078. Zero complete arrays are bitwise equal between CPU and GPU; such equality was not required.
- The fresh four-rank exact-zero control reaches20code-time units, residual maximum0 and final full ghost-inclusive metric exactly Minkowski. It performs801cycles due to a tiny final endpoint step. This is a target stop, not a walltime stop.

The test uses global16³ cells split into eight8³ blocks, original zero_rate/adaptedG1 and a Minkowski background. It establishes the repaired stencil’s tested ownership, response and CPU/GPU portability properties. It does not establish long-time perturbed-boundary stability or validate production stellar evolution.

Earlier8841975 failed during input validation because amplitude0 still selected family=lapse. Its records are preserved. Only the zero-family configuration was repaired, then all actual inputs were CPU-preflighted before8842005. No source, production campaign or monitor was modified by this replacement.

`results.json` retains compact numeric evidence. Full logs, histories, stage
metadata and individual checkpoint/characteristic validations remain in the
external record `outer-boundary-fix-20260920/gpu-stencil-verify-v2/results-8842005`;
the file hashes are retained in `provenance.json`. Signed binary stage arrays
and restart payloads remain in the corresponding Aurora run directory.
