# Aurora PVC qualification and science disposition

## Qualification

Job `8789659` ran on one Aurora PVC under `CompactBinaryMerger`, exited zero,
and wrote `AURORA_PVC_QUALIFICATION_PASS`. Eleven focused tests passed. The
evidence bundle is `evidence/aurora/qualification_8789659/`; its `SHA256SUMS`
digest is
`8acb38caa52c15236c3537e9d074bf76e9429409dc5120d66c7d75ce18e4ba27`.

The executable and CMake-cache SHA-256 values are, respectively,
`aae7ccb8739fb4951221ad7be69ea0e220548b52d402086f57d7857fa2c97a13`
and `8da40bcb47564d9184119ca207f9847a33a3d1b5bd2930627d705cda8fb36386`.

## Device regression

Job `8789460` bisected the PVC-only fault to commit
`151bf20d13c4838793f1c26e0b6b6e669cb7b765`. The fault was an implicit device
capture of host-owned `Z4c::opt` in the state-admissibility Kokkos lambda. The
repair commit is `f8303c6be7eb214fa1e91b646123ee0d434b3698`; it copies a scalar policy
before the lambda and adds a static regression contract.

## N256

Job `8789703` ran on one Aurora PVC under `CompactBinaryMerger`, exited zero,
and recorded `REACHED_TLIM` at coordinate time `30 M`. It is execution evidence
but not a successful Figure-3 qualification because the apparent first
curvature peak is constraint invalid. See `REPORT.md` and
`N256_REPRODUCTION.md`.

Jobs `8789668` and `8789692` were setup-only failures and produced no science.
No N128 or N512 replay was submitted after the N256 gate failed.
