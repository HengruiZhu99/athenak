# Aurora preflight v2: passed, four steps only

Job `8981414` passed all nine preflight cases on one Aurora debug node, using
12 MPI ranks and 12 distinct GPU tiles. The worker ran on 2026-10-29 UTC
(2026-10-28 PDT). These records establish native CPU/SYCL component parity,
short coupled evolution, and checkpoint restart correctness. **They do not
establish long-time stability, convergence, a wormhole-to-trumpet transition,
or production AMR.** No later job is assessed here.

The coupled evolutions use `n=16`, 77,824 active volume cells, 31 evolved fields,
a Cartesian core of half-width 0.125, and shells ending at 0.45, 0.75 and 1.
Minkowski, pulse and puncture reach coordinate `t=0.002` in four steps of
`dt=0.0005`. The puncture mass is `M=1/3`, so this is only `0.006 M`.
The original hyperbolic gauge is used with adaptive J disabled.

## What passed

| Cases | Verified result |
| --- | --- |
| `parity-device` | Native RHS CPU/SYCL maximum scaled difference 3.3973e-14; diagnostic difference 3.7467e-15. The component test uses 4,104 cells. |
| `parity-exchange` | Halo and sparse scri-jet CPU/SYCL differences zero; off-node null residual at most 1.1099e-15. |
| `parity-scri` | Null projection difference zero; scri rate scaled difference 7.8185e-14; dependent-D difference 1.8920e-14. Old-D/input-order independence passes. |
| `minkowski-short` | Four completed steps; maximum state deviation 1.1297e-14 and H L2 7.5309e-15. |
| `pulse-half`, `pulse-resumed`, `pulse-direct`, `restart-audit` | Direct four steps and two-plus-two restarted steps have identical full checkpoint SHA-256, including all four arrays. |
| `puncture-short` | Four completed steps with finite diagnostics, positive lapse/chi and a genuine FastFlow horizon solve. |

The CPU/SYCL comparison above is a **component comparison**, not a claim of
bitwise equality between full CPU and GPU trajectories. The restart comparison
is between two SYCL trajectories. All 4,920,320 checkpoint doubles, including
volume, surface and both compensated-RK arrays, agree bitwise. Both checkpoint
files hash to
`89400c9f365c093acb2a07451e2abdec2298a3eacec6e950c8f18d2bd4a11206`.

At the puncture endpoint, FastFlow finds mean coordinate radius
`0.06294908963512356`, areal radius `0.6666711388409639`, and mass
`0.3333355694204819`. Its expansion RMS is `4.4967418361e-4`, below the explicitly
chosen coarse-grid tolerance `1e-3`. This is a finite-resolution horizon, not
a roundoff-accurate solve. The legacy `hrms` column in the retained FastFlow
summary is mean-square; `horizon_expansion_rms` in the report is the RMS.

| Four-step endpoint | Minkowski | Isotropic puncture |
| --- | ---: | ---: |
| Exterior H L2 | 7.5309e-15 | 2.48393e-2 |
| Exterior M L2, Penrose norm | 2.8503e-15 | 5.75930e-2 |
| Maximum surface null residual | 6.6613e-16 | 8.8818e-16 |
| Maximum interpolated-ghost null residual | 8.8818e-16 | 1.2212e-15 |

Constraint norms use normalized compactified Cartesian coordinate volume.
`M_L2` uses the Penrose spatial metric; `M_physical_L2` includes the additional
Omega factor. For these spherical initial data, exterior norms use the
coordinate sphere at the mean FastFlow radius, rather than the full angular
horizon shape. The puncture acceptance checks require finite norms; they do
not establish their accuracy or convergence against 1D. Numerical-precision
nullness alone does not establish that all geometric scri regularity conditions
are satisfied.

## Provenance and verification

The frozen numerical sources match commit
[`113aaff783008204e1ff8cbf2fb8bb5f2af2204c`](https://github.com/HengruiZhu99/athenak/commit/113aaff783008204e1ff8cbf2fb8bb5f2af2204c)
on `codex/hyperbolic-core-shell`; Kokkos is pinned to
`6739bc623081648af9e752b616d9671527922cbf`. The numerical snapshot predates the
commit, which additionally includes documentation and local postprocessing.
The retained original provenance records that earlier uncommitted export;
the later immutable Git-object verification resolves its commit identity.

The full frozen source-manifest SHA-256 is
`3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f`.
The executable comes from the fresh `build-sycl-mpi-v4` directory; the MPI
binary SHA-256 is
`6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef`.
The [binding](binding.json) records component binaries, input, CMake-cache,
plan and launcher hashes. The [worker status](worker-status.json) retains all
nine commands, acceptance checks, artifact hashes and placement records.

The local packaging audit independently verified all 58 worker-bound artifact
hashes and byte counts, including the five checkpoint payloads, and reevaluated
all 94 report acceptance checks. All 1,110 AthenaK and 1,486 Kokkos source files
were compared with immutable Git objects at the stated commits. Plan, PBS,
source-manifest, rank binary/cache/log bindings and all 12 tile records also
match. See [verification.json](verification.json) and the concise
[assessment](assessment.json). Executables were not re-read remotely during
packaging; their identities are the historical bound hashes in those records.

The preceding job `8981303` remains a rejected preflight: a stale component
object lacked the mandatory backend marker after source extraction preserved
timestamps, so incremental make did not rebuild it. No physical evolution ran
there. A fresh v4 build directory compiled the same frozen sources for this
successful v2 preflight. The [v1 assessment](../preflight-v1/assessment.json)
is preserved.

This portable package contains small reports, histories, rank-zero logs,
placement/provenance records and FastFlow shape/summary output. Checkpoint
payloads and duplicate nonzero-rank logs were verified locally but omitted.
`package-files.json` inventories the retained files; it excludes itself.
