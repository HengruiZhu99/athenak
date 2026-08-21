# Cell-centered regression baseline

This baseline was captured before vertex-centered production edits.

- AthenaK commit: `6daa774d7451dbc5f7cac640c6e32a6fd11de7f9`
- AthenaK tree: `cbb702f4da954cf630da261790d5c21ef3142235`
- Kokkos commit: `6739bc623081648af9e752b616d9671527922cbf`
- Kokkos tree: `052310d99e85c4cfd72b6f561659c6b0a2aa19f3`
- Compiler: GNU C++ 13.3.0
- MPI: OpenMPI wrapper using `g++`
- CMake: 3.28.3
- Kokkos backend: OpenMP
- Problem generator: `built_in_pgens`
- Precision: double
- MPI: enabled
- OpenMP: enabled
- IrisK interpolator: enabled
- Executable SHA-256: `7a65f7be074ffa5fb006f2b4d8ddd00ed993d4d9470c2b22083e5e0a4a6044aa`
- CMake cache SHA-256: `4aff7c1166fdb53ae4bc697c3c9682ac4c69792cf28fecaa71db6fd6f27103c9`

The fresh build registered 65 tests. All 63 enabled tests passed in 184.17 s
with `OMP_NUM_THREADS=1 ctest -j16 --output-on-failure`. The two
CUDA-required tests were disabled in this CPU build:

- `athena.z4c_cartoon_production_kernels_cuda_required`
- `athena.pdf_scatter_cuda_required`

This establishes the host/MPI CC red-green baseline. CUDA/SYCL baselines and
the implicit-versus-explicit `grid_centering=cell` byte comparison remain open
until the immutable centering selector exists.

