# Native-VC dynamic-AMR page-fault root cause

## Proven failure

The retained pre-fix Aurora/SYCL canary entered the first accepted AMR
transaction and then raised a Level Zero `NotPresent` read page fault in the
coarse/fine boundary-rebuild interval. Stage fences narrowed the first bad
operation to `MeshBoundaryValuesVC::ProlongateVC`.

At the parent of commit `aa668dd0909eb8a144afe9ef398ae6e01e829b27`,
`src/bvals/bvals_vc.cpp` captured the following object in a device lambda:

```cpp
const auto coarse = recvbuf[n].iprol[0];
```

`recvbuf` and its `iprol` metadata are host-side `MeshBoundaryBuffer`
bookkeeping. The SYCL kernel therefore dereferenced a host pointer on the PVC
device. Host execution and CUDA's prior behavior were not proof that this
capture was portable.

## Repair

Commit `aa668dd0909eb8a144afe9ef398ae6e01e829b27` adds
`MeshBoundaryValuesVC::prolongation_bounds`, a
`DualArray1D<MeshBufferIndcs>`. `InitializeBuffers` copies each
`recvbuf[n].iprol[0]` record into the host mirror, marks it modified, and
synchronizes it to the device. `ProlongateVC` captures only
`prolongation_bounds.d_view` and obtains the range as `iprol(n)`.

The repair does not change the interpolation formula, the positive-chi gate,
neighbor classification, transfer inventory, or cell-centered code path.

## Dynamic proof

Aurora job `8775368` ran the repaired 2D Cartoon AMR canary twice through
phases A0--A19 and three times through phases A150--A159. It completed the
4-to-7-to-4 leaf lifecycle with no page fault. Its authenticated root is:

```text
/lus/flare/projects/CompactBinaryMerger/hzhu/
  z4c-vc-hardening-829be2f6-v1-20260822/
  campaign-device-metadata-repair-v2
```

The root `SHA256SUMS` digest is
`3c7a4f99a0032f63b8b6b80a814c3aa0a3a2fb5f4ff31ec46da3d91dcc6bce0f`;
its detached manifest digest is
`37a70671dc3c8679d9e6886b77e647ec1a66a3e10adb938de500643e0c8d9126`.
The exact
copies and complete digests are retained outside Git at
`/home/hzhu/Desktop/research/gr/collapse/artifacts/z4c_vc_hardening_20260822/aurora/campaign-device-metadata-repair-v2`.
The copied manifest uses immutable Aurora absolute paths; prefix-remapping only
that root to the local copy verifies every retained file.

## Separate unresolved failures

This repair closes the device-metadata page fault only. It does not establish
numerical AMR convergence. The retained nonconstant wave discriminator has
negative N16-to-N32 convergence in both 2D and 3D, and the exact-final SYCL
matrix exposes a separate 3D restart-continuation payload mismatch. Those
failures are reported independently and are not attributed to the repaired
host-pointer capture.
