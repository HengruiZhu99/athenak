# V4 GPU preflight: passed, epsilon 2 only

Aurora job `8983792` completed all fifteen planned cases on one debug node,
using twelve MPI ranks and twelve distinct PVC tiles. The worker took
123.803 seconds; PBS finished in state F with exit status 0, and the allocation
was released. Campaign-home usage at completion was 1,732,298,800 bytes.
This establishes a short implementation/restart preflight. **It does not
establish improved duration stability, convergence, AMR or a 100M puncture
transition.** The earlier V1/V2/V3 duration failures remain valid evidence.

[Canonical independent verification](verification-v2.json) rechecks all
221 plan checks, 131 worker-bound artifacts, 84 component rank logs, seven
native checkpoints, nine built binaries and all 2,613 source/Kokkos hashes
against immutable Git blobs. It also validates exact case order, generated
restart-input bindings, actual SYCL output, launch arguments and rank-to-tile
placement. No failed or missing cases were reclassified. The first verification
and its exact verifier are retained as `verification-v1.json` and
`verification-v1-verifier.py`; v2 additionally pins the V4 digest to its exact
executable name and requires a valid allocation-release timestamp.

The frozen source is `d98e2a99bbf1d5546956be3d724c82715ca704ec`; full source
manifest SHA-256 is
`0e7967f1f33559ed79b9f97d6416dacc296f0f14663d35c682177a9ac07db60f`.
The V4 MPI/SYCL executable SHA-256 is
`fc8d8074351a01ade6fd05dcf460555c935cf4ffe66c9bda31cc517679500fff`.
The reviewed plan and full build provenance are preserved under `raw/` and
`build/`. The prior [protocol](protocol.md) defines the acceptance scope.

All evolution cases use n16 and dt=0.0005. The following final diagnostics
are current at t=0.002, after four steps; they are not duration measurements.
Constraint norms use the code's normalized compactified Cartesian coordinate
volume. The puncture exterior uses a coordinate sphere at the mean FastFlow
radius, for this spherical-data comparison.

| Case | H exterior L2 | M exterior L2 | Surface null residual | Ghost null residual |
|---|---:|---:|---:|---:|
| Minkowski, epsilon 2 | 7.5299e-15 | 2.8689e-15 | 6.6613e-16 | 8.8818e-16 |
| Pulse, epsilon 2 | 5.9673e-4 | 6.5344e-3 | 6.6613e-16 | 8.8818e-16 |
| Isotropic puncture, epsilon 2 | 2.4830e-2 | 5.7678e-2 | 8.8818e-16 | 1.2212e-15 |

Minkowski's maximum state deviation is 1.0076e-14. Puncture determinant and
tracefree residuals are respectively 1.1102e-15 and 3.1086e-15. FastFlow found
a complete surface with coordinate radius 0.06294960072, mass 0.33333503543
and expansion RMS 0.00032517732, below the planned 0.001 tolerance, using
lmax=12, ntheta=24 and nphi=48. The puncture mass scale is M=1/3, so its
four-step endpoint is only t/M=0.006.

The active V4 configuration is epsilon=2, degree=7, with an 88-byte checkpoint
header. At n16 its reported conservative spectral bound is 25.1420849964
and maximum angular dissipation CFL is 0.402273359943, below 2.7. The seven
component cases include CPU/device, partition and reference parity. For the
new dissipation operator, maximum coefficient/reference error is 1.4780e-15,
source/reference error 4.5669e-14, energy error 3.6106e-15 and moment error
8.6302e-16; device and partition errors are zero. Additive closure CPU-rate
and dependent-trace errors are 8.4662e-14 and 2.0187e-14, with compatibility
2.7756e-17. These results validate the tested discrete implementations,
not the stability of a coupled trajectory.

The uninterrupted four-step epsilon-2 pulse and its two-plus-two restart are
identical in **every byte**, including the header and all 4,920,320 doubles.
The comparison covers volume state, volume RK compensation, scri state and
scri RK compensation. Separately, the V3 pulse and the V4 epsilon-zero pulse
have identical base-header contents except magic and identical four-array
payloads. Their whole files differ because V3 has a 72-byte header and V4 has
an 88-byte header with explicit epsilon=0, degree=7 configuration. This is
an off-control comparison, not permission for cross-schema restart.

The successful result makes **epsilon 2 alone** eligible for a separately
reviewed bounded duration plan at the same low resolution. Epsilon 0 is an
off-control here and lacks separate GPU restart qualification; epsilon 1 was
not evolved in this preflight. No admission is created by this verifier.

The package retains small reports, histories, rank logs, horizon sidecars,
plan/binding/worker receipts and build records. `raw/terminal-whitelist.json`
contains only job identity, terminal state, exit status and the original
collection hash; scheduler environment and account metadata are omitted.
All seven full checkpoints remain in the ignored local raw campaign directory
`output/hyperbolic-core-shell/aurora/preflight-dissipative-v1/`, and their
individual array hashes are retained in the verification. Rerunning the full
verifier requires that raw directory, the local build bundle, the exact input
file and the frozen Git/Kokkos objects. Fifteen focused offline tests cover
malformed inputs and identity failures; see `local-verifier-tests-v2.json`.
