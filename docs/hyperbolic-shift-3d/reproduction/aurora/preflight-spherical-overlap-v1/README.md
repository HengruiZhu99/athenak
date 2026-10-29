# V10 spherical-overlap GPU preflight

Job 8985881 completed successfully on Aurora, one debug node, twelve MPI ranks
and twelve GPU tiles, CompactBinaryMerger. All seven cases and 227 gates passed:
CPU/SYCL operator parity, short Minkowski, pulse direct/half/restarted, exact
checkpoint replay, and short isotropic puncture. Independent verification covers
39 artifacts and all 2643 frozen source-file hashes. See verification-v2.json.

Four-step n16 tests end at coordinate t=0.001 with dt=0.00025. Minkowski state
error is 6.02e-15, H norm 2.25e-15, M norm 2.85e-15. Puncture horizon mass is
0.33333365785 for M=1/3. Both null closures remain below 1.12e-15. These are
implementation checks, not duration stability or evolution convergence.

The first independent-verifier invocation rejected an inherited V9 input byte
count: the V10 input hash already matched the predeclared hash, but the new
geometry has more points. verification.json preserves that reader failure.
Only the reader's stale count was corrected from 23571224 to 34026590; source,
run plan, thresholds, input bytes, and results were unchanged. The final strict
reader and its 18 negative controls are in verification/.

Raw large checkpoints are archived locally under
output/hyperbolic-core-shell/aurora/preflight-spherical-overlap-v1. Source commit
77277bd25 and the two native binary hashes are recorded in build-proof.json.
Home campaign footprint at completion was 4,073,443,507 bytes, below 5 GB.
