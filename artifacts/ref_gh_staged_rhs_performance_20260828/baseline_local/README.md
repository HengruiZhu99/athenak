# Fresh local Phase-1 baseline

Source commit: `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`

The default-on source-unit log contains the unchanged deterministic analytic
coefficient, generated geometry, mixed moving-gauge, compact-boundary, and
all-61 RHS gates.  `evolved/` contains only compact logs and histories for one
compatible and one standard RK cycle with each reference backend.  The large
restart files were deliberately not copied from the temporary run directory.

Temporary reproducibility locations for this local execution:

- build: `/tmp/athenak-refgh-staged-baseline.QpPyb0`;
- source-unit run: `/tmp/refgh-staged-baseline-run.sdcd2Z`;
- evolved runs: `/tmp/refgh-staged-baseline-evolved.L81tJc`;
- regeneration A: `/tmp/refgh-staged-baseline-regen-a.qrJzWi`;
- regeneration B: `/tmp/refgh-staged-baseline-regen-b.d2wGxT`.

The first naive evolved-history comparison included the two backend-specific
reference-Ricci columns and therefore reported a false conditioned difference
of one.  The scientific comparison excludes only those two named diagnostic
columns, consistent with the accepted zero-generic-allocation architecture.
