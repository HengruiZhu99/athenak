# Common bulk diagnostic sampling control

The admitted C and D10M endpoints were each read with their own schema-correct diagnostic executable in a separate local SERIAL/MPI build, then evaluated using native, first4 and first6 outer bulk sampling. Original checkpoints and frozen Aurora binaries are unchanged. `run.py`, `commands.json`, source hashes in `validation.json` and build logs reproduce the six measurements. `analyze.py` reconstructs regional squared integrals and the inspected plot.

Native mode reproduces the original frozen diagnostic norms and radial integrals exactly for both C and D; explicit first4 equals native C and explicit first6 equals native D exactly. Invalid selections reject before opening a checkpoint. Checkpoint hashes match before and after all reads.

| Measurement | D/C exterior H change | D/C exterior M change | D/C .775–.9 H² change | D/C .775–.9 M² change |
|---|---:|---:|---:|---:|
| Each native operator | +0.772% | +36.553% | +3.577% | +114.604% |
| Common bulk first4 | +4.077% | +33.216% | +15.919% | +103.808% |
| Common bulk first6 | +5.658% | +32.133% | +24.201% | +97.084% |

Thus the increased outer momentum residual is not solely a difference in bulk diagnostic first-derivative order. D still has a larger measured exterior residual under either shared bulk choice. This does not establish the underlying continuum error or asymptotic convergence.

Scope: only outer-shell `PointKernel::Sample` first derivatives change for this diagnostic call. Native boundary refresh/halo construction and the donor-seven last-three-row derivative plan remain each case’s own; this is not a complete common boundary operator. Both cases retain identical second-derivative sampling and volume quadrature. No evolution call selects this override. The dominant .775–.9 region is reported separately; original native results remain preserved. An admitted E endpoint can be evaluated later with its schema-specific `native_shared_ko10_scri6` diagnostic; a failed E state must not be treated as a10M endpoint.
