# Boundary-refactor local equivalence

`pre/` is the local full-output prescribed-q run before the PVC boundary
projection split.  `post/` is the equivalent run after commit `a3d98182`.

For every trumpet, common-ADM, native Ref-GH, and user-history file, replacing
the run basename (`mb8_local` or `mb8_split_local`) by a common token produces
byte-identical files.  The final numerical rows are therefore identical.  The
restart files were about 16 MB each and are deliberately excluded from this
compact bundle.

This comparison establishes local equation-preserving behavior for the split;
it does not qualify the evolved PVC path.  Aurora job `8785718` still failed in
the first evolved cycle.
