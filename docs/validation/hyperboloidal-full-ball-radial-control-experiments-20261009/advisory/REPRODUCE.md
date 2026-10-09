# Reproduction and scope

The advisory checks were run in the original working directory
build-layer-research/continuum/full-ball-radial-assessment. Restore the exact
checker/runner bytes there before using their recorded commands; the scripts
resolve the workspace from that original depth. The immutable copy is an
archive and its paths are not rewritten to make it a new experimental run.

Use the Python executable pinned in receipt.json to run run_assessment.py.
This runs only exact solid-harmonic/weak-form identities and the small frozen
harmonic principal/projector/SAT algebra. The 288 actual principal rows are
read from the already frozen local gate, never regenerated. Its input index,
principal JSON hash and original source inputs are retained as small copies or
explicit size/hash metadata. No radial nodes, PDE differentiation matrix,
boundary solve, kernel rerun, eigensolve or evolution is constructed.

The initial structurally unequal comparison of equivalent (S-r)^2 factors is
retained under history/failed-structural-kin-assertion with the exact source,
tool-observed stderr and receipt. The corrected checker compares the simplified
difference to zero. It did not require a changed mathematical identity or a
relaxed tolerance.

ASSESSMENT.md, FINITE-RADIUS.md and VARIATIONAL-SAT.md are advisory. In
particular, finite principal energy-work identities do not establish a bulk
Gauss/SBP estimate, constraint-preserving boundary data, a closed continuum
generator domain or a uniform scri limit. The later black-hole objective still
requires a wormhole-to-trumpet interior with the Minkowski hyperboloidal
reference retained throughout; this regular-Minkowski polynomial discussion
does not admit a puncture basis or black-hole evolution.
