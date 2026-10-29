# Explicit baseline diagnostics: local snapshot validation

The MPI history and report now append `Theta_exterior_L2`, `Zr_L2`, `Zr_exterior_L2` and `horizon_irreducible_mass`. The first three use the existing globally reduced coordinate-volume sums; the last uses FastFlow's area-derived mass. `horizon_mass` keeps its Christodoulou definition. Old columns, equations, closure, solver settings and checkpoints are unchanged. This local addition is not in the frozen source-v10 Aurora preflight binary.

The V3 Cartesian/shell executable was rebuilt locally, then four target-zero snapshots were checked with `tst/hyperbolic_shift_3d/test_comparison_diagnostics.py`; no evolution step was taken. Each process was externally bounded to 410 seconds and completed in 6.2–8.2 seconds. The earlier reference snapshot used the immutable pre-change executable.

- The original n16 puncture's before/after checkpoint files are byte-identical, including all four arrays and the header: SHA-256 `a5405ccba10be5ce6e32e4dd575a5d31d37e999efe209c89cc0b18d3d0b5f66f`, 39,362,632 bytes.
- A nonzero fixture on the identity conformal metric sets Theta=.02(1−r²)² and Lambdaᶦ=.06 xᶦ(1−r²)², giving Z_r=.03r(1−r²)². P is offset by −2Theta, preserving physical K. The measured exterior Theta/Z_r RMS values are 0.00665192079604547 / 0.00479848422569499, agreeing with independently weighted analytic values to below 5e−16.
- One-rank and two-rank reductions agree to below 5e−16. Total radial Z is checked independently; history and JSON columns agree. Found horizons satisfy `horizon_irreducible_mass = horizon_areal_radius/2`.
- The same fixture with expansion tolerance 1e−15 intentionally leaves the finder unconverged at the fixed 300-iteration cap. Exterior Theta/Z_r/H/M_r and irreducible mass remain CSV NaN/JSON null, while total radial Z remains finite. This is unavailable horizon measurement, not evidence of horizon disappearance.

The analytic source, source hashes, complete small reports/histories/logs, commands and [assessment](assessment.json) are retained. Raw fixtures, executables and checkpoints remain local and are omitted from this package. The original input and coordinate-file hashes were unchanged. This checks reporting, reductions and snapshot state equivalence, not trajectory stability or convergence.

Example reproduction from the repository root (fresh output directory required):

```sh
python3 tst/hyperbolic_shift_3d/test_comparison_diagnostics.py \
  --binary build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_projected_angular \
  --initial output/hyperbolic-core-shell/puncture-v3/n16/initial.txt \
  --coordinates output/hyperbolic-core-shell/puncture-v3/n16/mesh.coordinates.csv \
  --reference-prefix output/hyperbolic-core-shell/comparison-diagnostics-local-v1/before-puncture \
  --output output/hyperbolic-core-shell/comparison-diagnostics-repeat-new
```

The reference prefix must name a preserved pre-change target-zero run with the same input and finder settings. No scheduler job or standard-Z4c evolution is used.
