# Candidate 1 Aurora PVC discriminator 8790496

Status: **FAIL**.  No performance benchmark was launched.

Commit `6d098aa84f97abefc27b4777bb4e64153a36092a` compiled with the compact
q-reduction destination in `SYCLDeviceUSMSpace`; the compiler report confirms
that memory space.  The one-rank initialization, initial RHS, output,
diagnostics, and initial q reduction passed.  The first evolved-stage q
reduction then faulted with a PVC `NotPresent` device write immediately after
`CopyU`, before its debug fence.  Eight-rank execution was not attempted.

This demonstrates that merely moving the result destination off HostSpace does
not make Kokkos's custom 32-Real reduction portable for this allocation layout.
It does not show nonfinite Ref-GH state or a hot-reference coefficient failure.

Remote campaign and retained large build:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_candidate1_qdevice_7335058c`

