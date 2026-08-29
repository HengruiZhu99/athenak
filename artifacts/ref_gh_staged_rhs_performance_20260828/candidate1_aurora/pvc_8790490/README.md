# Candidate 1 Aurora PVC build gate 8790490

Status: **FAIL AT BUILD**.  No PVC cycle or benchmark ran.

The fresh debug-queue job used commit
`7335058c6d0963eb75eaa217456b113f7058adc9`.  The equation-preserving
device-result q reduction compiled in the non-MPI local build, but the Aurora
MPI build rejected a `const array_sum::GlobalSum` as the writable receive
buffer of `MPI_Allreduce`.  The retained compiler diagnostic is exact.  The
qualifier was subsequently removed without changing the reduction algebra.

Remote build and campaign location:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_candidate1_qdevice_7335058c`

