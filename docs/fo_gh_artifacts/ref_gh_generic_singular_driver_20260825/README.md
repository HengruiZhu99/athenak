# Ref-GH generic singular-driver artifacts

This directory contains compact local evidence from the first implementation
checkpoint.  It does not contain field dumps or restart files.

## Evidence

- `local/exponent_estimator_strict.log`: expected nonzero hard-gate run.  The
  first-order estimator passes analytic pointwise checks; direct FD fails to
  converge on the prescribed shell.
- `local/generic_reference_scan.log`: source-unit and staged-cache oracle log.
- `local/generic_reference-generic-reference-scan.tsv`: dynamic and static
  maxima for all requested tau, Gaussian-width, and resolution combinations.
- `local/generic_reference_scan_analysis.json`: model fits and prescribed-q
  verdict produced by `scripts/ref_gh/analyze_generic_singular_reference.py`.

## Exact local commands

```text
cmake --build build-feedback-local -j 8

./build-feedback-local/src/athena \
  -i tst/inputs/ref_gh_generic_singular_estimator.athinput \
  job/basename=docs/fo_gh_artifacts/ref_gh_generic_singular_driver_20260825/local/exponent_estimator

./build-feedback-local/src/athena \
  -i tst/inputs/ref_gh_generic_singular_reference.athinput \
  job/basename=docs/fo_gh_artifacts/ref_gh_generic_singular_driver_20260825/local/generic_reference

python3 scripts/ref_gh/analyze_generic_singular_reference.py \
  docs/fo_gh_artifacts/ref_gh_generic_singular_driver_20260825/local/generic_reference-generic-reference-scan.tsv \
  --output docs/fo_gh_artifacts/ref_gh_generic_singular_driver_20260825/local/generic_reference_scan_analysis.json
```

## Local provenance

- Parent/source SHA: `0e248310a562c8a84327421eecf70f2f5d1da4a3`
- Branch: `codex/ref-gh-generic-singular-driver-20260825`
- Kokkos SHA: `6739bc623081648af9e752b616d9671527922cbf`
- Compiler: Ubuntu GCC 13.3.0
- Build: Release, `-O3 -DNDEBUG`, Kokkos Serial
- Executable SHA-256:
  `95e1c53ff423c13c1d93820ce789a7746279cd719a163fc1463a4da9b47ba512`

This is local CPU evidence only.  It is not CUDA or Aurora PVC qualification.
