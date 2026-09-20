# Eight 32³-block CPU MPI smoke test

PASS for the requested bounded block-size/execution check. This does not clear vacuum perturbation stability or stellar production.

The final immutable active-stencil CPU MPI executable has SHA256 `67ff395e3c7af43425ef13fe4f0ebecef00256db9450f6ac912f5ab39620f1f8`. The exact earlier oblique Minkowski fixture was changed only to global64³ / eight32³ blocks on the same [-1,1]³ domain, with heavy balance/signed-stage dumps disabled and per-rank restarts enabled. All six faces use original zero_rate CPBC with linear ghosts. The unchanged fixture has G1, kappa1=kappa2=eta=lapse-residual-damping=0, sixth-order volume differences and RK3; it is not the production matter/damping configuration.

Four independent fresh starts ran exactly three RK3 cycles to0.01875M: zero and amplitude0.01 oblique lapse pulse on one/four MPI ranks. The zero input disables both characteristic pulse family and amplitude. The pulse keeps center0.75M, width0.5M and transverse width1M. Invalid-state/metric/CPBC checks remain active. No source or production changes were made.

Each final cohort contains eight32³ blocks with40³ ghost-inclusive storage and12,800,000 residual entries. All rank headers match within a cohort, all raw checkpoint payload values are finite, and full Minkowski-plus-residual lapse/chi and all Sylvester metric minors are positive including ghosts. Every zero-case residual entry is exactly zero. Both pulse cases retain11,263,991 nonzero entries, with a genuine off-diagonal raised-normal response at physical/internal block edges. The complete residual block payloads, including all25 fields and ghosts, match bitwise between one/four ranks separately for each case. Comparisons use the same eight-block layout; no bitwise claim across different block sizes is made.

MPI1/4 execution times were12.65s/4.07s for zero and13.03s/4.08s for the pulse. Total generated output is about1.084GB, below the4GB cap. Initial and final per-rank restarts, logs and exact input files are preserved. `results.json` records input/executable/script/checkpoint hashes, per-block residual hashes, finite/SPD minima and comparisons. `summary.json` omits individual file/block hash lists for compact review.

Reproduction (use a new output directory):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_smoke.py \
  --repo /path/to/athenak-outer-boundary-fix \
  --exe /path/to/immutable/athena-active-stencil-mpi \
  --output /new/block32-smoke-directory
```

`--analyze-only` rechecks existing outputs without evolution. The script imports the repository's existing oblique-input builder and checkpoint/cohort validation functions, records its explicit input and executable hashes, rejects unexpected checkpoint layouts, and enforces the output-size cap. Snapshot dumps are disabled, so this test certifies the final checkpoint and checked histories, not saved per-stage states or long-time stability. No Aurora job was submitted.
