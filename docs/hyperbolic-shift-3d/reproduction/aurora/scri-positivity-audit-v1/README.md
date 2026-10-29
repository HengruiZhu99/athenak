# Read-only audit of the original scri positivity rejection

The three saved failed stages reject an **angular-interpolated conformal metric with a negative determinant** near a cubed-sphere face corner. Every stored scri metric and every metric in the implicated 36-point donor stencil remains positive definite. All 31 interpolated fields are finite and χ is positive. This localizes the immediate rejection; it does not establish which operator first excited the preceding growing mode.

The captures are from the unchanged original SYCL v4 executable, numerical source `113aaff783008204e1ff8cbf2fb8bb5f2af2204c`. The associated [resolution comparison](../resolution-control-comparison-v1/README.md) establishes their growth histories and outcome distinctions. No projected or cross-face candidate is included here.

| Case | Accepted rollback time | dt | Failed interpolant det(g) | Smallest eigenvalue | χ at query | Minimum donor det(g) | Rollback det(g) at same query |
|---|---:|---:|---:|---:|---:|---:|---:|
| n16 original | 0.4895 | 0.0005 | −0.00644909 | −0.00329861 | 0.660949 | 0.752424 | +0.0478692 |
| n24 | 0.313666667 | 1/3000 | −0.0124108 | −0.00591952 | 0.895148 | 0.979400 | +0.0688607 |
| n16 half dt | 0.486 | 0.00025 | −0.0232697 | −0.0107086 | 0.692266 | 0.776921 | +0.0455364 |

Times are coordinate time. The headerless failed-stage capture does not record the RK substep or its abscissa. The table time belongs to the unadmitted checkpoint's rollback state; the attempted intervals are respectively [0.4895, 0.4900], [0.313666667, 0.3140], and [0.4860, 0.48625]. The comparison samples the saved rollback surface at the same coordinates and does not restart the checkpoint or alter its admission flag.

At each failed query the first and second leading principal minors remain positive, but the determinant is negative. `Geometry` therefore returns before computing its inverse, leaving `code_q=0`; that zero is a consequence of the invalid metric, not a separate negative inverse-normal contraction. Across every other valid-metric query, q stays positive. No query fails χ positivity or finiteness. The minimum eigenvalues over all stored surface metrics are 0.566650, 0.588197, and 0.596948 respectively.

| Case | Boundary Cartesian direction | Donor outer block / face | Local angular (u,v) | Target ghost block; (j,k) | Sum of absolute donor weights |
|---|---|---|---|---|---:|
| n16 original | (−0.554035, +0.588662, −0.588662) | 15 / +y | (0, 0.0294118) | 14 / −x; (16, −1) | 27.8386 |
| n24 | (+0.561644, −0.585045, −0.585045) | 16 / −y | (0, 0.98) | 13 / +x; (−1, −1) | 24.6762 |
| n16 half dt | (−0.554035, −0.588662, −0.588662) | 16 / −y | (0, 0.0294118) | 14 / −x; (−1, −1) | 27.8386 |

Indices are zero based. The implicated radial ghost indices are n, n+1, n+2. There are two distinct binary64 query coordinates per case, differing by about 10⁻¹⁶ because radial normalization is evaluated at each ghost depth; these represent one physical direction to 12 decimal places. Counts are from global exact-coordinate deduplication (6,616 total queries at n16 and 12,938 at n24), not the sum of MPI rank-local counters.

These directions lie on a face edge near a corner. Both angular coordinates are outside the rectangle covered by that face's cell-center samples: u=0 lies below 1/(2n), and v is just below 1/32 at n16 or just above 47/48 at n24. The existing componentwise six-by-six angular polynomial consequently extrapolates there. Its weights sum to approximately one but include values down to −4.82 and up to +5.79. Positive definite donor matrices do not guarantee a positive definite result under these signed weights. This is evidence for the direct interpolation failure, without claiming that replacing interpolation alone will stabilize the evolution.

The [C++ helper](scri_interpolation_audit.cpp) reproduces the original `BuildHalos` boundary-coordinate selection and `HyperbolicCompositeExchange` donor indices, weights, anchor-difference summation, packed metric reconstruction, and rejection predicate. It invokes the actual `CompositeMesh` and `Geometry` implementations. It reads only the failed surface portion; it never calls evolution, null projection, or a restart loader. The [Python runner](run_audit.py) checks six relevant source headers against the frozen commit, verifies all six input payload hashes and sizes both before and after replay, and independently reconstructs the interpolants and metric eigenvalues with NumPy. This is a CPU binary64 replay, not a bitwise re-execution of the SYCL kernel. The observed negative determinants are far larger than floating-point rounding differences.

Four temporary constant-field controls passed: positive identity geometry gives no rejection; negative χ gives only the χ rejection; an indefinite metric gives only the metric rejection; a nonfinite auxiliary gives only the finiteness rejection. All three scientific replays completed in under 0.03 seconds each after a bounded local build. The six input files remained byte-for-byte unchanged. The original n16 failed-stage hash comes from a post-collection remote hash supplement, explicitly distinct from the original worker artifact list. The other five inputs are worker-bound. [provenance.json](results/provenance.json) records the exact authorities and hashes.

Run from the repository root, with NumPy and the configured local `build-hyperbolic-mpi` Makefiles build available:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/scri-positivity-audit-v1/run_audit.py \
  --output output/hyperbolic-core-shell/aurora/scri-positivity-replay-new
```

The output directory must be new. `--raw-root` changes the parent containing `stability-v1/output` and `resolution-control-v2/output`; `--build-dir` and `--compiler` select the local build and compiler. Raw failed-stage/checkpoint payloads must be supplied separately at their recorded hashes; they are not included in this small package. The runner refuses differing frozen source headers or input bindings. Compilation is bounded to 120 seconds and each diagnostic subprocess to 410 seconds. No remote action is performed.

[results/summary.json](results/summary.json) contains both failed queries per case, eigenvalues, rollback comparisons, and clock limitations. The three detailed case JSONs retain all 31 interpolated fields, donor minima/maxima, exact coordinates, donor indices and all 36 weights. `controls.json`, build/execution records, and `inventory.json` complete the small reproducible evidence package.
