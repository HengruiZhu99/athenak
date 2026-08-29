# Candidate 1 Aurora PVC discriminator 8790472

Status: **FAIL**.  No performance benchmark was launched.

The fresh one-node debug-queue job built commit
`91c43bc6420218cc61d5718d90e40db9e256f88e` and proved eight distinct PVC
tile mappings.  Its one-rank initialization, initial analytic RHS, boundary
operations, diagnostics, and initial compact q reduction all completed.  The
first evolved stage then stopped immediately after `CopyU` with a PVC
`NotPresent` write fault in the compact q reduction.  The eight-rank cycle was
therefore not attempted.

The compiler diagnostic identifies that reduction as writing its result to a
`HostSpace` destination.  The failure was exposed after adding the 141-Real
hot view; it is not evidence of a nonfinite physical state.  The retained
`one_rank/run.log`, compiler report, mapping, provenance, and histories are the
complete compact evidence.  Large executable/build objects remain only under:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_candidate1_91c43bc6`

