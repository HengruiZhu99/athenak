# Additive half-step projection readback (HELD)

This is a fresh source candidate awaiting root review and exact authorization.
It does not change the original driver, attempt, saved radial matrix, kernel,
scientific witnesses, stencils, fields, or thresholds. No new compilation,
generator spectrum, or propagation is included.

The original attempt `../attempt-1791558113840144000` remains failed. It stopped
at the sixteenth case, constant lapse at r=.96 on the x axis, because the
continuum-rate last increment was 1.8406042311e-6 rather than at most 2e-7.
Its observed orders were 4.0143, 4.0030, 4.0072; the final and extrapolated
gauge-zero tests passed. Its full five-level records and all query outputs are
retained unchanged. This new run cannot retroactively pass that attempt.

The sole scientific refinement is

    old h0 = min(.002, (.98-r)/4)
    new h0 = old h0/2
    five levels h_j = new h0/2**j, j=0,...,4.

All fourteen witnesses, twenty-one centers, physical-eight field ordering,
the same polynomial-interpolant comparator, fourth-order Cartesian stencils,
five sequences plus bulk/total defects, Richardson formula, and every original
threshold are unchanged. In particular a sequence still needs a non-unresolved
order classification and last scaled increment <=2e-7; gauge initial zero
is <=5e-11 and final/extrapolated continuum zero is <=2e-7. Bulk/SAT/total
constraint magnitudes are measurements and are not required to vanish.

The previous source map is reused only at exactly identical binary64 Cartesian
point tuples. Its full 150-column outputs are loaded from hash-bound saved
query/call files; query schema, channel order and three WJet basis directions
are independently checked. New points alone are queried using the unchanged
Release executable and full source-batch schema. The same 504 direct arbitrary
WJet controls at all original centers are queried afresh and compared against
the combined map over all 150 columns at the unchanged 5e-11 threshold. Input
and output algebraic normals keep separate input/RHS local denominators. The
actual eight-row constraint-rate API is queried for every new case/refinement
level as before. No coefficient map is interpolated spatially. A cache-reuse
receipt gives exact reused/new counts and ordering; all original cache files
are included in before/after source hashes.

The exact original attempt receipt, call index and direct-linearity receipt
are pinned in the new driver. The saved NPZ raw input/source map is an additive
postprocess of those same original outputs, and remains independently useful;
the refined driver reads the original full-width logs to preserve the original
150-column direct-linearity check.

Execution remains held until root writes a separate authorization binding this
driver, its plan, the unchanged executable 2293e9be..., and the unchanged
J0/N8/rb=.98 operator 2ed0da45.... Use the saved radial assembler environment:
CommandLineTools Python, `PYTHONPATH=build-layer-research/boundary/python-deps`,
`OPENBLAS_NUM_THREADS=1`. Floating warnings and exceptions remain errors.
Every failure must be retained without in-place edits or tolerance relaxation.

This is a finite-dimensional projection/SAT constraint-defect readback at
transition/collar sample points, not a CPBC, physical-energy, Einstein-closure,
continuum instability, or later black-hole formation result. The required
single-BH wormhole-to-trumpet transition with the Minkowski hyperboloidal
reference remains outside this local gate.
