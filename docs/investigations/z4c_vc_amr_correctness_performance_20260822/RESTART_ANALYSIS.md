# Restart analysis

## Baseline failure

The base branch had a non-bit-exact 3D SYCL continuation despite matching
hierarchy, time, and timestep.  The accepted-state discrepancy appeared after
restart reconstruction, but the baseline evidence did not uniquely distinguish
shared-copy ordering from coarse-cache/dependent-boundary reconstruction.

## Current result

After the floor-index, derived-halo, buffer-cardinality, and deterministic
finest-authority repairs:

- Release/OpenMP 2D Cartesian restart: exact payload pass;
- Release/OpenMP Cartoon restart: exact payload pass;
- Release/OpenMP 3D Cartesian restart: exact payload pass;
- ASan/UBSan/Kokkos-debug restart construction and teardown: clean after
  repairing the independently detected ownership leaks;
- Aurora/PVC 2D Cartesian restart: exact pass;
- Aurora/PVC Cartoon restart: exact pass;
- Aurora/PVC 3D Cartesian restart: exact pass.

The test resumes checkpoints immediately before refinement, while refined, and
after derefinement.  It compares the final global binary payload against the
uninterrupted run, validates immutable centering/storage metadata, and rejects
legacy CC-to-VC reinterpretation before allocation.

An exact payload match implies zero absolute, relative, and ULP difference for
all serialized accepted-state values, including all 25 evolved Z4c components.
It does not independently compare transient padding or every ghost/coarse-cache
slot that is reconstructed rather than serialized.

## Interpretation

Observation: the prior 3D SYCL mismatch is absent on the repaired source.

Inference: deterministic finest-level authority and coherent derived coarse
halos are the most plausible reasons, because those are the repairs that affect
restart reconstruction ordering and dependent data.

Limitation: the available before/after data do not isolate one of those changes
as the sole cause.  This report therefore does not label the earlier mismatch
as a proven backend nondeterminism bug.

## Remaining restart gates

- Current-source CUDA restart and memory checking are pending.
- Multi-rank 3D SYCL rank-change job `8775888` was submitted but its result was
  not retrievable after authentication failed; this row remains open.
- A current-source physical Brill restart/replay has not been run.

The post-cleanup full debug restart wrapper was not completed: it reached the
outer six-minute orchestration timeout without a sanitizer report.  This does
not invalidate the completed Release exact-payload tests or the clean direct
sanitizer restart/teardown, but it is not counted as a completed debug restart
matrix.
