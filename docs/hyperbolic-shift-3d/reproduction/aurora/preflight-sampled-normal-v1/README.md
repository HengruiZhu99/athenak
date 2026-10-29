# V5 sampled-normal Aurora preflight — submitted

Job **8984104** was submitted to CompactBinaryMerger/debug on one node, with
12 MPI ranks and GPU tiles. Its last observed scheduler state was queued.
This package records the reviewed plan, exact bindings and successful fresh
build; it contains **no GPU test result or scientific admission** yet.

Scientific source: `9d72318c14a2099b5aea118cca62e9fbd8773350`.
Source-manifest SHA256: `8dd9936710dafbdd145f5ee95ba4d48da47e295afff805b7f3f380e0afd15b36`.
The 2,624-file source includes the preserved 1,486-file Kokkos dependency.
The new source-v12/build-sycl-mpi-v12 directories preserve all previous exports.
The fresh SYCL/MPI build used make -j64, passed in 425.06 seconds, and left the
campaign at 1,889,286,308 bytes, below its 5,000,000,000-byte cap.

Seven cases cover GPU component parity, four-step Minkowski, pulse direct and
restarted evolution with exact checkpoint comparison, and a four-step puncture.
The puncture finder uses lmax12/ntheta24 with the existing RMS tolerance .001.
The V5 normal scheme and its 104-byte checkpoint configuration are explicit.
Passing these bounded tests would qualify only the planned low-resolution
duration control, not sustained Minkowski stability or a 100 M puncture result.

The subsequent proposed comparison starts independently from analytic
Minkowski at n16, dt=.0005, epsilon2, targeting coordinate t=.6. It requires
separate review of the actual preflight evidence before admission/submission.
It retains the existing state-deviation limit 1e-8 and H/M limits 1e-7.
The sampled boundary closure has roughly third-order manufactured accuracy.
