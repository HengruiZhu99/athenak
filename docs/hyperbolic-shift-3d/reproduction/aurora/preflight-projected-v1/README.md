# Projected candidate: bounded preflight passed

All **12 preflight cases passed** on one Aurora debug node with 12 SYCL MPI
ranks and 12 distinct GPU tiles. This validates component comparisons,
four-step evolution, and restart handling for the optional final-stage
algebraic-projection candidate. **It does not establish duration stability,
a wormhole-to-trumpet transition, or convergence.** The candidate remains
deferred while resolution controls using the unchanged original driver are
prioritized.

The compiled numerical source is frozen **source-v5**, commit
[`fdcb9f44f32aad9028ffc76d51113a9927d97a35`](https://github.com/HengruiZhu99/athenak/commit/fdcb9f44f32aad9028ffc76d51113a9927d97a35).
The separate **launcher-v6** comes from
[`3c8fa2451b3dc41207fdba10c4010f847d87ea45`](https://github.com/HengruiZhu99/athenak/commit/3c8fa2451b3dc41207fdba10c4010f847d87ea45).
The bound combined source-manifest SHA-256 is
`8d0eba426c84cdf9997864cf7471a44914cc590b9a4d4ea2efe022b944287c93`.
The projected MPI binary SHA-256 is
`362093fc8c998576c7d88a14afdc2c5ebcdcb6a68f408b7fdaf92ad62b7b1486`.

The candidate normalizes the conformal metric determinant and removes the
conformal-curvature trace at the accepted final RK stage, then reapplies
null closure and dependent D. Its checkpoint schema is **HYP3D_MPI_V2**.
Adaptive J remains disabled. This preflight uses the original same-face scri
angular stencil; the later cross-face angular prototype is not included.

## Validation results

| Case | Result and scope |
| --- | --- |
| `parity-device` | Native CPU/GPU bulk and partition comparison passed; worst scaled RHS error 3.3973e-14, diagnostic error 3.7467e-15. |
| `parity-exchange` | CPU/GPU halo and sparse scri-jet differences zero; off-node null residual reduced from 1.5615e-2 to 1.1099e-15. |
| `parity-scri` | Native rate scaled error 7.8185e-14; dependent-trace error 1.8920e-14; old-D input and evaluation-order checks passed. |
| `parity-algebraic` | Tensor, invalid-input rejection, GPU/CPU, unchanged-field, compensation and scri-reclosure checks passed on every tile. Surface/ghost null residuals <= 1.1103e-15 after reclosure. |
| `benchmark-staged`, `benchmark-inline` | All 12 distinct owned surface partitions passed; timing comparison below. |
| `minkowski-short` | Four steps to coordinate t = 0.002; stored-field deviation 1.0051e-14, H L2 7.5297e-15. |
| `pulse-half`, `pulse-resumed`, `pulse-direct` | Two steps to t = 0.001, restart for two more, and independent four-step direct run to t = 0.002. |
| `restart-audit` | Direct and resumed V2 checkpoint payloads bitwise equal across 4,920,320 doubles; maximum absolute/scaled differences zero. |
| `puncture-short` | Four steps to t = 0.002 = 0.006 M; accepted horizon mass 0.333335569420, coordinate radius 0.0629490896351, expansion RMS 4.4967e-4. |

The evolution cases use n = 16 per block, 77,824 active volume cells and
dt = 0.0005. Their final diagnostics are current, and their V2 checkpoints
are admitted for this scheme. In the puncture case, H exterior L2 is
0.02483017035 and surface/ghost null maxima are 8.8818e-16 / 1.2213e-15.
These are early-time measurements, not a passed longer stability campaign.
The Minkowski and pulse coordinate times have no black-hole mass scale.
The [assessment](assessment.json) and complete [case reports](output/)
retain the precise values.

## GPU scri sampling microbenchmark

At n = 16, each rank evaluates **128 distinct surface rows** out of 1,536,
with 25 sampled fields. The staged variant caches field jets in a separate
sampling kernel; the inline variant performs the same sampling within the
native tensor kernel. Each uses three warmups and 20 timed `Evaluate` calls,
with explicit Kokkos fences around the timed interval. The staged jet cache
uses 332,800 bytes per rank.

| Per-rank mean time per `Evaluate` | Staged/cached | Inline |
| --- | ---: | ---: |
| Mean across 12 ranks | 3.7201934 ms | 47.6400437 ms |
| Minimum rank mean | 3.7018171 ms | 46.6351858 ms |
| Maximum rank mean | 3.7303220 ms | 48.4253385 ms |

The **ratio of these rank means is 12.8058**; paired inline/staged ratios
range from **12.5439 to 13.0068** across ranks. These are per-rank `Evaluate`
microbenchmark ratios. There is **no inter-rank timing-start barrier**; the
fixed order is staged followed by inline, and ranks share GPU package
resources. Setup, state upload and warmups are excluded. `Evaluate` includes
its per-call host summary reduction, but the benchmark excludes MPI
communication and coupled evolution. The result therefore does not measure
whole-node throughput, allocation wall-time improvement, or coupled
evolution speedup.

Every paired rank has exactly matching recorded `rhs_norm2` and
`surface_norm2`, and each variant reports zero repeat error. **Equal norms
are not a bitwise full-array cross-variant comparison.** Each variant
separately checks that its full output remains unchanged across repeats;
the native CPU/GPU component comparisons and the bitwise projected-restart
comparison above are separate validation evidence.

The complete 12-rank records are retained in
[staged](output/benchmark-staged.report.json) and
[inline](output/benchmark-inline.report.json) reports. A compact
[paired-rank CSV](benchmark-ranks.csv) and
[comparison summary](benchmark-comparison.json) can be regenerated with
`python3 summarize_benchmark.py` from this directory. The script verifies
report hashes, rank/tile identities, row ownership, timing consistency and
recorded norm equality; it launches no tests or evolutions.

## Evidence binding and omissions

[verification.json](verification.json) independently rechecks all **97
worker-recorded artifact hashes and byte counts**, including the five
checkpoint payloads omitted from this package. All **130 acceptance-check
records** reproduce and pass. It also checks plan/PBS bindings, the
checkpoint headers, all **1,113** entries in the retained non-Kokkos source
manifest against the numerical commit, and all **five** launcher-manifest
entries against the launcher commit. The Kokkos pin is retained in
[source-v5-provenance.json](source-v5-provenance.json).

The combined remote source manifest is identified by its bound hash; its
full payload was not included in the local download. The retained
[numerical-source manifest](source-v5-non-kokkos-manifest.json),
[launcher manifest](launcher-v6-manifest.json), and provenance files make
their Git source identities independently inspectable.

All per-rank metrics remain in the reports. Raw component and benchmark logs
are retained for rank 00 only; the verification record retains hashes for
the omitted rank logs. The package contains no executable, full checkpoint,
or raw scheduler/account-balance records. `package-files.json` inventories
the retained files and excludes itself. This local package makes no
terminal-scheduler claim; candidate duration tests remain deferred.
