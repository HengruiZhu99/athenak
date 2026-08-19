# Delta-upper source-repair evidence

This directory is the compact local evidence for the two-pass construction of
`Delta^A_BC` in the reference-covariant GH source. It was run from a clean
detached worktree based on parent `a2706e33745f2d6d7d1452e1ae8cda9f35acbc03`,
with the source-repair changes in the commit that includes this record.

The executable was a Release/Kokkos-OpenMP build with 32 host threads. The
root checkout could not be used for a fresh build because it has unrelated
user changes in the boundary-value code; none of those paths are included in
this evidence or staged by this repair.

Commands, substituting the clean-worktree path for `<worktree>`, were:

```
OMP_NUM_THREADS=32 <worktree>/build-openmp/src/athena -d <out> -i <worktree>/inputs/ref_gh/ref_gh_source_unit.athinput
OMP_NUM_THREADS=32 <worktree>/build-openmp/src/athena -d <out> -i <worktree>/tst/inputs/ref_gh_minkowski.athinput
OMP_NUM_THREADS=32 <worktree>/build-openmp/src/athena -d <out> -i <worktree>/tst/inputs/ref_gh_linear_wave.athinput mesh/nx1=N meshblock/nx1=N
OMP_NUM_THREADS=32 <worktree>/build-openmp/src/athena -d <out> -i <worktree>/tst/inputs/ref_gh_stability.athinput mesh/nx1=N mesh/nx2=N mesh/nx3=N meshblock/nx1=N meshblock/nx2=N meshblock/nx3=N
```

The stationary `t=0` ladder used the committed static-trumpet input with 64,
96, and 128 active cells per MeshBlock. It performed no time evolution. The
complete captured terminal records and compact histories are retained beside
`summary.tsv`; no large simulation output is committed.

The later `nonflat_source_unit_openmp.txt` uses 128 deterministic manufactured
stationary references with a nonzero time-space coframe component, spin
connection, and Riemann tensor. It evaluates the production C++ covariant
source against the retained C++ coordinate source and scalar-frame transform.
The coordinate derivatives of all frame components are initialized before
reconstructing the coordinate metric jet; this matters because their first two
array indices are frame indices, not coordinate metric indices.
