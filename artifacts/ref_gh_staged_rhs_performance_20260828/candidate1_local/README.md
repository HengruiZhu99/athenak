# Candidate 1 local qualification

Code under test: `ba563692e7e0c541eae35c94e35b33ee29a1336e`.

This is the first selective-cache discriminator: 12 static plus 8 stage
analytic coefficients, a 141-Real symmetry-reduced hot view, and the qualified
loop-form covariant GH source.  The monolithic generated source and the generic
1171-Real pipeline remain independent oracles; neither is allocated or called
by the analytic production task path.

The full current generator was executed twice with SymPy 1.14.0.  Both output
sets are byte-identical to the committed geometry, gauge, source, and hot-cache
headers.  The source-unit log retains the unchanged coefficient, expanded
radial, full geometry, mixed moving-gauge, boundary, and three-way all-61 gates.

`evolved/` contains compatible and standard one-cycle analytic/generic runs and
the deterministic comparison JSON.  The maximum positive-time conditioned
difference is `2.9721190786258234e-13`, in the source-curvature diagnostic;
common ADM histories agree to `6.54e-15` or better.  The gate is the unchanged
`5e-12`.  The analytic allocation is 12 + 8 + 141 Reals per ghosted cell and
zero generic bytes.

No restart or field dump is committed.  Temporary full local locations were:

- build: `/tmp/athenak-refgh-staged-baseline.QpPyb0`;
- final source unit: `/tmp/refgh-hot-all61-final.BOa89D`;
- final evolved runs: `/tmp/refgh-hot-evolved-final.vYVf5g`;
- second deterministic regeneration: the path recorded in
  `/tmp/refgh-staged-candidate1-regen.path`.

Aurora PVC qualification and performance retention remain pending.
