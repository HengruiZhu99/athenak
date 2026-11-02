# Completed SG1 comparison at 10M

All three original SG1 resolutions completed10M. The two N32 checkpoints and
all recorded worker artifacts are now locally length/SHA256 verified. The
completed remote checkpoint copies were retired only after that preservation;
see ../full-artifacts-verified.json and ../verified-retirement.json. Frozen
science/builds remain intact. No new SG1 evolution was run.

Native matching SG1 checkpoint diagnostics reproduce the full-domain reported
norms and use one common exterior cutoff r>0.1099, outside all three final
candidate horizons. History curves still use each run's own candidate horizon;
accepted mass histories retain gaps rather than filling rejected finder rows.

| Grid | Exterior H RMS | Exterior momentum RMS | Exterior radial momentum RMS | Accepted final mass error | .775–.9 fraction of exterior H² / M² |
|---|---|---|---|---|---|
| N16 | 0.01541497 | 0.02900764 | 0.03081597 | +0.34133% | 62.20% / 75.02% |
| N24 | 0.00820183 | 0.02550556 | 0.02610776 | -0.02868% | 81.65% / 93.72% |
| N32 | 0.00685511 | 0.02456275 | 0.02489983 | -0.03603% | 95.08% / 98.79% |

Effective norm orders using actual spacing ratios are H=1.556,
0.623 and momentum=0.317,
0.131 for N16→24 and N24→32. These are not asymptotic
fourth-order convergence. The N32 exterior squared residual is concentrated in
.775–.9 (95.08% H²,98.79% M²), while the .16–.30 contribution decreases strongly.
This strengthens the motivation for the separately controlled restoration/KO/
scri changes; it does not identify their effects before those runs exist.

The radial plot includes the interior as well as the exterior; large inner
values are excluded from the exterior norm. Its coarse radial bins are retained
for comparisons, not interpreted as pointwise maximum locations.

Reproduce with ../../short-gauge-v1/compare_resolutions.py,
passing the shared raw mirror and N16/N24 segment01 plus N32 segment01/02 in
order. The exact diagnostic commands, checkpoint hashes, binary hash, norm
values and counters are in comparison.json. A fresh output directory is required.
