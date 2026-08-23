# Backend and MPI matrix

## Build provenance

| backend | source | Kokkos | status |
|---|---|---|---|
| GNU 13.3 OpenMP host | local worktree | `6739bc623081648af9e752b616d9671527922cbf` (4.7.2) | pass |
| Intel oneAPI SYCL/PVC | Aurora | same Kokkos commit | numerical matrix pass except any explicitly open row below |
| CUDA | Perlmutter | not built on current source | pending: SSH certificate rejected before login |

Aurora current implementation executable at source commit `21b91213`:

```text
8b90b70d649d203d7cb40e9e2cb177438d541e224ee512bb4eec21d5293b3149
```

Aurora rebuilt executable at test/provenance commit `cb4f173b`:

```text
f89a8495a87f186abc68f2a9c8992ee384151b73af3dbc09a352470248e4dbab
```

Aurora rebuilt executable with distributed migration repair `480de5f7`:

```text
80b59c0cbe6cd04f283366493e6052a1ba819260dfef345096c79e8a79a29bf1
```

The CMake cache hash is
`82ed2949f7d7eb693e0350d9a5542c32780c315f002c299e46dca37566767018`.
The runtime reports Kokkos default execution space `SYCL`, Level Zero device
selection, oneAPI 2025.3.2, MPICH 5.0, and Intel Data Center GPU Max 1550 PVC.

## Runtime matrix

| case | host/OpenMP | Aurora SYCL/PVC | Perlmutter CUDA |
|---|---|---|---|
| dynamic AMR O2, 2D/Cartoon/3D, 1 rank | pass | pass | pending |
| dynamic AMR O4, 2D/Cartoon/3D, 1 rank | pass | pass | pending |
| dynamic AMR O6, 2D/Cartoon/3D, 1 rank | pass | pass | pending |
| static multilevel O4, 2D/Cartoon/3D | pass | pass after loading `py-h5py` | pending |
| gauge wave O2/O4/O6, 2D/3D | pass | pass after loading `py-h5py` | pending |
| output, 2D/Cartoon/3D | pass | pass | pending |
| restart, 2D/Cartoon/3D, 1 rank | exact pass | exact pass | pending |
| Cartoon O4, 2 ranks | local MPI unavailable | pass | pending |
| Cartoon O4, 4 ranks | local MPI unavailable | pass | pending |
| 3D O6, 2 ranks | local MPI unavailable | pass before distributed repair | pending |
| 3D O6, 4 ranks | local MPI unavailable | pre-repair page fault; repaired job result unavailable | pending |
| 3D rank-change restart | local MPI unavailable | repaired job result unavailable | pending |

The first Aurora wrapper returned status 8 because nine analysis scripts could
not import `h5py`; all numerical runs that did not require that module passed.
After loading `py-h5py/3.14.0`, all nine previously blocked tests passed.  This
was a harness dependency, not a source or device numerical failure.

An initial four-rank 3D attempt was invalid because the fixture created only one
root MeshBlock.  Athena rejected it before evolution.  Commit `cb4f173b` makes
that selected test use eight physical root blocks; this is a fixture correction,
not a numerical relaxation.

Job `8775888` was submitted on the `480de5f7` executable to rerun repaired
two/four-rank 3D O6 and a rank-change restart.  Its final scheduler state and
outputs were not collectible after the Aurora login certificate was rejected.
Submission is not runtime evidence, so those rows remain open.  Commit
`144aaded` only changes host ownership cleanup and does not supply a substitute
device result.

## Qualification boundary

`VC_Z4C_BACKEND_QUALIFIED` remains **NO** until current-source CUDA runtime,
CUDA restart, and CUDA memory-check evidence exist.  SYCL results cannot be
used as a proxy for CUDA.
