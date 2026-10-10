# Held outer mass-measure product diagnostic

The actual v9 radial readback remains failed. Its exact traceback identifies
`measure * (bilinear(y,hy,angles) + bilinear(u,u,angles))` at verifier line431;
progress last records index608 of769. The exact failing index is unknown. Fix
the diagnostic window to609..640, justified by that progress checkpoint and the
prior division site627, without asserting onset at any particular index.

Reconstruct only the two already-rounded mass bilinears and their sum at these
32 radii, using the exact original modal recurrence, H action, layout, einsum,
tested weighting helper and BLAS operand/reduction path. Observe the one outer
measure product with strict NumPy flags. Classify all32*64*64=131072 scalar
products using the unchanged conservative possible-tiny predicate; only possible
tiny products receive exact binary Fraction multiplication and the previously
tested nearest-even rounding helper. Keep array/scalar underflow flags, counts,
first/last affected radius, exact local error, zero/subnormal classifications and
up to64 bit examples. No E accumulation, local arithmetic replacement or gate
upgrade is performed. Ordinary flags stay strict; any other failure is retained.

The minimal operand registry is explicit: four selected arrays in the pinned
radial operator NPZ (`source_coefficient_radii`, `radial_weights`,
`angular_weights`, `source_reference_rows`), the corresponding read-only window
of `input/output.bin`, and the frozen J0 orbital-L layout JSON. No E/K/G/SAT,
coefficient actions, source tables, reference queries, SVD or generator arrays
are loaded. The NPZ is opened lazily and only those four names are accessed.
Selected NPZ decoding, input-map reads, NumPy/SciPy imports and Fraction targets
remain held until exact parent authorization and independent source review.

Source preparation records byte/AST equality of the copied original operand
functions and exact H/modal/bilinear statements, complete original source/runtime
pins, original actual failure and independent association. All input payloads
are metadata only in this source checkpoint. Future attempts are unique, with
full outer command/environment/stdout/stderr/returncode and pre/post hashes.
No compiler, query, native evolution, spectrum or propagation is part of this
diagnostic. Any eventual correction requires a fresh separately reviewed source.
