# Same-data N10 scri node-count control, 2026-10-27

No coupled case was rerun. This diagnostic uses the saved **actual** N10
`t=.0001` midpoint of the completed related three-resolution family. It retains
all 50,000 volume cells, all 600 surface points, the same 22 stored fields, the same
saved shift warm start, the same boundary fields and algebraic null/fixed-angle
shift datum. The physical equations and all sampler guards are unchanged.

The seven-node local midpoint maximum is `5.06665186395e-8`, LambdaZ at
row 533, face 5, j3,k3:
`x=(-.276172385369497,-.276172385369497,-.920574617898323)`.
This local value is **not** the reported `1.35403992276e-7` maximum over the
completed evolution's stage/accepted source evaluations. The saved midpoint does
not contain the fields at every later maximum. It cannot identify the component
and location of that later maximum by itself.

Eight nodes are **unsupported by the existing guard**:

```
Hermite angular/radial factor-condition guard
```

The seven-node sampler's maximum factor indicator is `1.64921328579e7`.
An independent direct infinity-condition computation on the scaled logical
radial Vandermonde gives radial conditions `70222.5740741` and `547207.603704`
for 7/8 nodes. Angular weights are unchanged, so the predicted eight-node maximum
factor indicator is `1.28514521436e8`, above the predeclared `1e8` guard. The
constructor stops as designed. No guard was relaxed, no lower-degree fallback
was substituted, and no unguarded eight-node source was reported. A condition
indicator exceeding this engineering guard is not a proof of mathematical
ill-posedness, but it prevents accepting this configuration under the existing
qualified sampler contract.

The seven-node LambdaZ numerator decomposes as follows. Normal here means the
coordinate-radial derivative of global Cartesian components; the unchanged
physical kernels perform their metric contractions.

| Contribution | Pole numerator |
|---|---:|
| Metric normal first jets | +5.69178175844e-7 |
| Metric angular first jets | -5.74596682332e-7 |
| Independent Lambda | +5.85217846681e-10 |
| Trace/Theta normal first jets | -1.82852174848e-7 |
| Trace/Theta angular first jets | -7.40324963246e-10 |
| Curvature contraction with grad Omega | +1.37759269812e-7 |
| Sum | -5.06665186395e-8 |

The sum agrees to `1.2e-22`. All six A-tensor components also retain separate
baseline, chi normal/angular, metric normal/angular, independent Lambda and
curvature contributions at every point. Across all rows, decomposition error is
below `3.56e-15`; the first-jet pole helper agrees with the independent full
ConformalRHS tensor pole within `1.34e-15`. The comparable metric-normal/angular
terms and their cancellation remain visible. They do not justify imposing zero
normal derivatives or projecting the independent evolved connection.

## Independently recomputed original vector residual

The checkpoint's beta array is explicitly named `shift_warm_start`. The
classical `Step` computes its final beta on the last RK4 intermediate state,
then combines the accepted metric/lapse/curvature/boundary fields without another
shift solve. Therefore that saved warm start is not a guaranteed algebraic
solution for the accepted midpoint fields. The restart correctly begins its next
stage with a fresh solve, and the completed family has bitwise restart agreement.

A fresh original-operator application using the accepted midpoint geometry/source
and the **unchanged saved** beta gives:

- physical RMS `6.51203743966e-6`;
- conformal RMS `8.85982400224e-6`;
- point maximum `7.47256197684e-5`.

These fail the original solve tolerances for this deliberately unsolved accepted
state/warm-start pair. They do not contradict the recorded passing original
residuals at the actual stage solves. No solve was performed to hide the mismatch.
This bulk operator/source computation is identical for both requested sampler
node counts: only the boundary jet sampler was being changed. It also preserves
the full original curvature source rather than manufacturing `L beta` data.

Seven-node local clock/null/tangent residuals are 0,4.44e-16,3.62e-16. Frozen
volume, shift and surface arrays compare exactly unchanged after the diagnostic.
No time advance, shift solve, projection, initial-slice replacement, or physical
boundary/source modification occurred.

## Reproduction and conclusion

The standalone executable was compiled from the exact completed-family source
capsule and private Kokkos dependencies, plus the new diagnostic source. Compiler
dependency proof has zero live project paths. Build took 4.46 seconds; the
read-only diagnostic took 6.42 seconds, each under an owned 180-second process cap.
Evidence, failure reason, source/binary hashes, all-row lossless CSV.gz, and full
component maxima are in
`evidence/related-scri-nodes-20261027/` and
`evidence/related-scri-nodes-analysis-20261027/`.

```
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/run_related_scri_nodes.py \
  --family docs/elliptic-shift-3d/evidence/related-scri-family-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-related-nodes
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/analyze_related_scri_nodes.py \
  --input docs/elliptic-shift-3d/evidence/related-scri-nodes-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-related-nodes-analysis
```

There is no admissible eight-node improvement result, so no further coupled run
is supported by this control. Localizing the later failed maximum requires its
actual stage/surface data or explicit component/location diagnostics on a future
authorized run. The completed three-resolution family remains scientifically
failed; the below-target midpoint value does not override that result.
