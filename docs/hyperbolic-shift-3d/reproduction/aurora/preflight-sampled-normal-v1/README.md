# V5 sampled-normal Aurora preflight — passed

Job **8984104** completed with scheduler state F/exit0 on one CompactBinaryMerger
debug node. Independent verification passed all **167 plan gates**, all **39
worker-bound artifacts**, twelve distinct GPU tiles and the exact restart.
This qualifies only the named low-resolution implementation for duration
testing; it is not a sustained-stability, convergence or 100 M result.

## Results

| Check | Result |
|---|---:|
| CPU/SYCL numerator scaled error | 5.67e-16 |
| CPU/SYCL surface-rate scaled error | 6.96e-14 |
| CPU/SYCL dependent-D scaled error | 1.75e-14 |
| Simulated ownership difference | 0 |
| Finite-D numerator dependence | 0 |
| Native default versus frozen kernel | Bitwise equal |
| Four-step pulse versus two-plus-two restart | Complete files bitwise equal |
| Minkowski maximum field deviation at t=.002 | 1.20e-14 |
| Minkowski H/M RMS at t=.002 | 7.53e-15 / 2.82e-15 |
| Puncture horizon expansion RMS at t=.002=.006M | 3.25e-4 |
| Puncture horizon mass | .3333350354 |
| Maximum evolution surface/ghost null residual | 1.11e-15 |

All seven cases completed. The puncture uses lmax12/ntheta24/nphi48 and the
unchanged .001 expansion-RMS tolerance. Its exterior H/M coordinate-volume
RMS is .02483/.05768 on this coarse initial grid; these short results do not
establish agreement with the 1D baseline or puncture convergence.
The checkpoint comparison covers all four state/compensation arrays and
the full 104-byte V5 header. Native null closure uses the current geometry.

## Frozen implementation and scope

Scientific source: `9d72318c14a2099b5aea118cca62e9fbd8773350`.
Source-manifest SHA256: `8dd9936710dafbdd145f5ee95ba4d48da47e295afff805b7f3f380e0afd15b36`.
The new source-v12/build-sycl-mpi-v12 directories preserve all previous
exports. All 2,624 source blobs, including 1,486 Kokkos files, were verified.
The fresh MPI/SYCL/PVC binary64 build used make -j64 and passed in 425.06s.
The V5 evolution binary SHA256 is
`08fee8c5c9b5fa1815f52ec63e968bba75dbf04762dba991bf359c2ecb3c078d`.
Campaign storage after this job was 2,072,107,926 bytes, below the 5 GB cap.

Only the geometric normal-pole derivative is replaced by the derivative of
sampled native numerators. Independent Lambda/Theta, bulk equations, weak
penalties and angular epsilon2 remain. A separate surface-owner halo fill
precedes sampling; final dependent D and volume ghosts are rebuilt afterward.
The manufactured boundary accuracy is roughly third order. Passing component
and restart tests does not settle the coupled evolution's stability.

The coordinator separately issued [admission.json](admission.json) after
reviewing [the independent verification](verification/verification-v1.json).
It covers V5 epsilon2 at n16, and permits a separate fresh Minkowski duration
control targeting t=.6 at dt=.0005. It does not admit scientific stability.

## Reproducibility

Small raw reports, histories, rank logs, horizon sidecars, job receipts and
build records are retained here. Full checkpoints and the initial field file
remain in the ignored local raw directory
`output/hyperbolic-core-shell/aurora/preflight-sampled-normal-v1/`.
The verification records all array and full-file hashes. Scheduler environment
data are omitted; terminal-whitelist.json retains the original collection hash.
The verifier is copied byte-for-byte from its executed staging version; its
default paths reflect that original location. To rerun this portable copy,
provide explicit paths from the repository root:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/preflight-sampled-normal-v1/verification/verify.py \
  --repo "$PWD" \
  --raw-dir output/hyperbolic-core-shell/aurora/preflight-sampled-normal-v1 \
  --build-dir output/hyperbolic-core-shell/aurora/preflight-sampled-normal-v1/build-v12 \
  --initial-data output/hyperbolic-core-shell/aurora/preflight-sampled-normal-v1/inputs/puncture-n16.txt \
  --output output/preflight-sampled-normal-repeat.json
```
