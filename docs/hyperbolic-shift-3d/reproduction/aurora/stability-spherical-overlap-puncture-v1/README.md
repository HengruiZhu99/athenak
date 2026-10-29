# V10 puncture stability pilot — failed: negative chi near the puncture

Aurora job 8986048 was submitted after the independently verified V10 preflight.
One debug node, twelve MPI ranks/GPU tiles, CompactBinaryMerger; immutable
source-v18 is commit 77277bd25. The plan targets coordinate t=1 (3M) from fresh
isotropic wormhole data, n16, dt=0.0005, with the existing full 3D equations.
There is no lapse floor, Lambda reset or enforced spherical symmetry.

Plan SHA256: 303e93cd821bafe79f0ff2aad89f29a042718df15d8fccd87f46f76816784b24.
Admission SHA256: 2f87e55d379be509a37a6a5646ed9a840817d02f2b60511bcf5ec151ae8112eb.
2001 steps are expected: 2000 full steps plus the roundoff-sized final interval
from the existing clock loop. The full step remains 0.0005.

Do not treat submission or an admitted checkpoint alone as scientific success.
Collect the terminal job, release the campaign allocation lock, archive and
verify all artifacts and replay the predeclared gates. The independent duration
reader in ../../evolved-overlap-v1/verify_duration.py accepts an explicit pinned
V10 preflight decoder and admission. No result or long-time claim is recorded here.

The matching Minkowski t=1 plan is prepared locally at
output/hyperbolic-core-shell/aurora-staging/stability-spherical-overlap-minkowski-v1-plan.json
(SHA256 2b78a72613f549f7891c40b54009b0f46e10d68a717c72420116b23b667b9ffe)
and remotely at the campaign root, but is not submitted. Do not overlap allocations.

## Collected result

Terminal F, exit 1, 199 accepted steps, coordinate t=0.0995=0.2985M.
The native error is invalid input at active cell 4095: eight chi values became
negative, while lapse remained positive and all state values were finite.
The last accepted checkpoint is unadmitted and cannot be resumed. Independent
verification passed the evidence audit and reports scientific_checks_passed=false.
All artifacts were archived and the allocation lock released. See
../../continuous-positive-v1 for the diagnosis and separately versioned candidate.
