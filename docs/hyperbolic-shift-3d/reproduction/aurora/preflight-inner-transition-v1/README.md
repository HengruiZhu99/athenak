# V7 GPU qualification passed; duration stability remains unproven

Aurora job **8984332** completed all seven cases on one CompactBinaryMerger/debug
node with twelve MPI ranks and twelve distinct GPU tiles. An independent
verification replayed **184 checks**, checked **39 worker-bound artifact hashes**,
verified every checkpoint array and exact configuration, and confirmed terminal
PBS stateF/exit0 plus allocation release. This is **implementation/restart
qualification only**, not long-time stability, convergence or100 M acceptance.

| Test | Result |
|---|---:|
| Complete CPU/SYCL bulk RHS maximum scaled error | 3.3615e-14 |
| CPU/SYCL scri RHS maximum scaled error | 7.8994e-14 |
| Dependent surface trace maximum scaled error | 1.9270e-14 |
| Simulated surface partition difference | 0 |
| Finite-D dependence of pole numerators | 0 |
| Actual twelve-rank direct versus2+2 restart | whole checkpoint bitwise identical |
| Short Minkowski t=.002 state deviation | 8.8159e-15 |
| Short puncture t=.002=.006 M horizon mass | .333335035428 |
| Short puncture horizon expansion RMS | .00032517734 |
| Maximum recorded evolution null residual | 1.1103e-15 |

The V7 component test additionally exercises complete device-generated inner
halo data and its coupled bulk RHS. Both agree with the CPU oracle within
3.3615e-14. Offline reconstruction of the actual GPU puncture checkpoint agrees
with native whole/exterior H/M norms within1.46e-16 absolute error, explicitly
using V7 interpolation. This is diagnostic parity, not evolution accuracy.

The immutable source-v14 commit is
`9022f51f26e66acafdd3190d0c33f0c1ea53a661`; all2637 source/Kokkos files were
verified against Git blobs. Its source-manifest SHA is
`da9304ba17ad5a1336aef4468dc88ce375622104bac2c69cc1d3cc534abcb2a4`.
The precise-FP64 SYCL/MPI build used `make -j64` and bound all three named
executables before submission. The reviewed plan SHA is
`a2cdb0ff507a7edc632cb2ce1c226cf50cd30562a7d729e7abc537583995c7d8`.
Campaign footprint after qualification was **3,378,457,541 bytes**, below5GB.

V7 changes grid halo interpolation in the inner transition while retaining
V6's scri discretization, equations and independent fields. It uses distinct
magic `HYP3D_MPI_V7` and tuple(3,7,4,3). The [preflight admission](admission.json)
qualifies only that exact configuration for subsequent tests. It does not
assert that the interpolation control cures every mode or the V6 puncture
lapse failure. The [local native evidence](../../inner-transition-native-v1/README.md)
includes one/two-rank ownership and restart controls.

Build metadata is in `build-receipts`; independent verification and its16
negative-control tests are in `verification`. Small raw output is retained
under `raw`; large checkpoint arrays and full collection metadata remain in
`output/hyperbolic-core-shell/aurora/preflight-inner-transition-v1`.
To repeat the full verifier from the repository root, pass explicit `--repo .`,
`--raw-dir output/hyperbolic-core-shell/aurora/preflight-inner-transition-v1`,
`--build-dir output/hyperbolic-core-shell/aurora/preflight-inner-transition-v1/build-v14`,
`--initial-data output/hyperbolic-core-shell/puncture-duration-v3/n16/initial.txt`,
`--pins` pointing to this package's `verification/pins.json`, and a fresh
`--output` JSON path to `verification/verify.py`.
