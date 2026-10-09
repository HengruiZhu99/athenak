# Exploratory null-feedback source snapshots

These files preserve an ignored private experiment, not a production option.
The candidate's original-reference native run failed near t=1.35; its wider
reference run reached t=2 with growing constraints. See
[the follow-up report](../../hyperboloidal-stability-followup.md).

`immutable-manifest.json` records the original read-only snapshot hashes and
large local artifacts. `native-build-receipt.json` records source identities,
compiler/configuration flags, the forced include and all native translation-unit
commands. `source-before.json` agrees with the implementation runtime source at
`27c19d20`. The later tangent-bin fix does not change the native equations.
Copied source files retain their original bytes for hash verification, including
one whitespace-only line in `wide_audit.cpp`; they are archival experiment
sources outside the production build.

`candidate_injection.hpp` loads the production gauge and redirects the native
physical-P calls through `null_feedback.hpp`, explicitly assembling added
shift poles. `candidate_cmake.cmake` applies that overlay only to the private
Athena target. Recorded paths refer to the original ignored scratch directory;
restore that layout or adapt absolute paths when reconstructing the experiment.
Never mix this executable with the clean production runtime receipt.

`fourier_audit.cpp` and its Python checks regenerate the full local 20-field
Fourier matrices, phase checks, constraint residues and zero-wave pole limits.
The summary JSONs retain checks and selected high-frequency modes. Full local
matrices/reports are retained by hash rather than committed. Frozen local roots
do not establish a global continuum or native discrete spectrum.

The preserved `native_candidate.md` was written before the long runs completed;
the final native JSON report and follow-up report retain their negative outcomes.
The finite-Q nonlinear closure counterexample remains valid.
