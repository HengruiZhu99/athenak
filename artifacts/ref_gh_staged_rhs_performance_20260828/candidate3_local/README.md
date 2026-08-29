# Candidate 3 local qualification

Code under test: `627b0d447c39c67e364dda36ca2fcbc1e83e30f8`.

This is the Phase-13 coordinate-source A/B discriminator, activated only after
candidate 2 passed correctness/PVC with low scalar-kernel spill but remained
arithmetic-bound at `12.7851x` the matched Z4c RHS.  It preserves the standard
GH equation and replaces only the production algebraic representation:

1. the 32-Real physical pass also packs the coordinate metric, derivative,
   implicit reference-gauge derivative, gauge source, and constraint;
2. ten flat work items evaluate the coordinate symmetric partial-wave source;
3. ten flat work items apply the analytic frame transformation;
4. the unchanged componentized Pi principal update consumes the transformed
   source.

The transient allocation is 126 Reals per active cell: 32 physical, 84
coordinate/transform, and 10 coordinate partial-source values.  This is below
the hard 128-Real cap and is a measured exception to the 106-Real target:
candidate 2's low-spill covariant scalar kernel alone consumed 54.08% of the
complete PVC stage.  Persistent analytic storage remains 12+8+141 Reals per
ghosted cell and generic-cache allocation remains zero.

The unchanged source-unit gate passes all 216/2160 coefficient, 2376 geometry,
2160 moving-gauge/dtTheta, 2160 boundary, and 4320 compatible/standard all-61
checks.  The componentized coordinate source agrees pointwise with the generic
covariant oracle to operation-conditioned error `4.83589e-16`; the complete
all-61 maximum remains `2.84217e-14` against `256 epsilon`.

Near the most singular expanded sample (`r/M=0.03`, `q=2`) the unconditioned
coordinate/covariant difference is about 7.23%.  This is the previously known
coordinate cancellation, not hidden by a relaxed gate: the committed oracle
uses the generic source operation condition already used by the unchanged
all-61 comparison, and both raw implementations remain available.  The PVC
full-output evolved gate must still reject any resulting nonfinite or divergent
physical behavior.

Fresh prescribed-q compatible and standard one-cycle production runs agree
with the independent generic-cache backend to conditioned history Linf
`2.9721190786258234e-13`, below `5e-12`.  Analytic allocation for the 16-cubed
test is 1,048,576 physical bytes, 2,752,512 coordinate bytes, and 327,680
partial-source bytes.  Restart files and field dumps are excluded.

This is local correctness evidence only.  It does not qualify PVC execution,
compiler pressure, or performance.
