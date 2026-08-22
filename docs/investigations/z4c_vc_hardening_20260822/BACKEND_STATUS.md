# Backend status

## Host and MPI

- Release OpenMP+Serial: full enabled suite passes.
- Debug OpenMP with Kokkos bounds and DualView checks: all completed tests are
  clean; the VC-specific 3D record/replay test passes in isolation.
- ASan+UBSan OpenMP: build and selected 2D/3D lifecycle, restart, output, task
  order, and invalid-target matrix pass with leak detection disabled; no
  address or undefined-behavior diagnostic appears.
- Release MPI+OpenMP: exact production source builds successfully. The local
  launcher is not a trustworthy runtime authority; actual rank/device tests
  are run on Aurora.

Exact local hashes:

| Build | Executable SHA-256 | CMake cache SHA-256 |
|---|---|---|
| Release OpenMP+Serial | `fe7f1d8e314df3b8a574a8e7cb6a1fb3ca4948dc9a7c0cd1240abb08d68cc29b` | `7d7e8d7c4ed28e8e3af5047fa5930fd746ac124822fb6470b1dee12532e6cf5e` |
| Release MPI+OpenMP | `584b3e3bbf40f3e22da634fe50846bdcb0bb1efb85cc31dc1e571d79926e7731` | `847bb3de956fc0f9eff08b115ad96abbdb0f796b0e7a2bcc26f0cdb1751d26b6` |
| Debug bounds/DualView | `0969669e979b4c5c3aab495b52bc2298e27133add4a540f4889b137c17a13634` | `c321803e14350e43468aff70dfceea959641440d609241e9fe8121418854fcd6` |
| ASan+UBSan | `e3176533d13279f31c211c48442004d010f0dbc9656f50af94b9d73667e549cd` | `519898ac59912f663290013ff3d15e38993718e3ea14f535c3e6387eccafa9c3` |

## Aurora PVC / SYCL

The exact compiled production authority is commit
`99a4eb5ba7713f7de73239cf75a27c1fb9ac6cbb`, tree
`e8c1083cc9ea67aa4a3a2c3adbffb9c31fe32c83`, with executable
`48c257bac9ed0802805661ca93630eb526082e1adfdaebc485c9d11fab1b4ff1`
and cache
`621b7caa192fa36d651583c127dd8d3aac01766211f351f63dfcfd5cc992892d`.
Aurora job `8775503` built successfully and ran 21 selected tests:

- all nine O2/O4/O6 VC refine/derefine lifecycle cases in 2D Cartesian,
  2D Cartoon, and 3D Cartesian pass on PVC;
- 2D Cartesian and Cartoon restart pass;
- all 2D/3D output cases pass;
- six gauge-wave tests were not exercised because the configured Python lacked
  NumPy (harness environment, not a numerical failure);
- 3D Cartesian restart continuation fails exact final-payload equality after
  the post-refinement checkpoint.

The two compared 3D restart payloads have equal length (`7,861,236` bytes) but
different SHA-256 values:

- fresh final: `e79600535d266b13b06fefec6aec2b80d51354bf0b5875e8533c4dd29a073217`
- post-refinement restart final: `131e6929ef1397916b191d6963c02561f0f107f615dabd5493cb2614cdc26e69`

There are `2,134,655` differing bytes. Histories agree in hierarchy, time, and
timestep, but differ at roundoff-scale in constraints and final `max_abs_K`.
This is a real exact-restart qualification failure, not evidence of a large
physical-state divergence. Aurora job `8775539` added NumPy, but its Python
still lacked `h5py`; all six gauge-wave harness invocations failed before data
analysis and the fail-fast script never reached restart or MPI. That job is
recorded as a harness failure, not a numerical result. A final narrow job,
`8775545`, ran only the two-rank 2D Cartoon O4 and 3D Cartesian O4 lifecycle
cases. Both pass on PVC. Its immutable manifest digest is
`7b7e038e39361f3b55d8bdfc300490c72053de4a83505c33e57151ad3faca485`;
the detached-manifest-file digest is
`efd4cb6e4e9b71533805fc60a6b7799e36338a17574d19f10aecf875948cfcc3`.

## CUDA

The historical exact-final CUDA build exists, but a runtime matrix for the
current repaired production source has not been completed. Perlmutter rejected
the available expired certificate during this run. No CUDA pass is claimed.

## Backend verdict

The exact-final two-rank SYCL cases pass, but the 3D SYCL restart failure and
absent current CUDA runtime gate prevent backend qualification.
The conservative overall verdict is therefore `VC_Z4C_NOT_QUALIFIED`.
