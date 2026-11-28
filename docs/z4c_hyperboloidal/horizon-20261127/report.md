# Hyperboloidal Z4c performance on Horizon

The CUDA compatibility baseline `b943db139` built on Horizon and evolved the exact N12 binary to cycle 130. Job 109722 ran for 1,028 s on one four-GB200 node, for **0.5711 SU**. Two N12 jobs cover timing, profiling and formulation checks. The two-node N24 seed depends on their success. Their outstanding wall-time limits amount to 6.3333 SU. [Job ledger](ledger.tsv)

## CUDA build and correctness

Unmodified `c80c114e4` failed to compile because the extended device lambda in `EvaluateBlocks` first captured `ind` inside `if constexpr`. Job 109457 was cancelled before it acquired a node. The compatibility commit adds `ind` to the existing no-op capture before the branch. The Release build uses Kokkos 4.7.2 at `6739bc623`, CUDA 13.3.73, the Blackwell 100 target, relaxed constexpr, Open MPI 5.0.11 and METIS. Floating-point contraction is disabled on both host and device. [Compatibility commit](https://github.com/HengruiZhu99/athenak/commit/b943db139a3ba094aaf31f4470ef4d6712705ebb)

On macOS arm64 with Kokkos Serial, the compatibility build and `c80c114e4` produced byte-identical radial 1D outputs in both evolution modes through 22 steps. In a separate single-rank, 25-block 3D fixture, both complete restart files, history, and horizon diagnostics matched through two cycles in each mode. Every one of the 5,826,600 stored doubles per checkpoint was finite. The false-mode radial output also matched `85e3256fc`. The optional radial profile explicitly writes NaN for the undefined endpoint Hamiltonian and momentum.

The same 27 of 28 CPU CTest cases passed in both builds. In `hyperboloidal_reference_accuracy`, the baseline geometry residual was `6.67702e-08` against a `1e-09` limit, and the constraint residual was `7.86191e-11` against a `1e-11` limit. The gauge residual passed. The test also failed with Release flags and with contraction explicitly disabled.

## Benchmark inputs and measurements

These are the spin-0.5 near-head-on decks from the Aurora evolve-all run directory. Their identities match the manifests of the existing Horizon initial-data exports.

| Deck | Input SHA-256 | Initial-data identity |
|---|---|---|
| N12 | `cad95d8dac125bffd6ced079dc6ea2c79bff4afc1ec18534aa41ffdab21a0309` | `7d32c45f889a8c6f224eae2441653dc3cd859e39a904922145968cdbb8c650c2` |
| N24 | `b286b2f90b2176e20f1aea1f067ba1ca84102559d9a3e3137941a144196a9234` | `223431d81d0a89cdca6f88c412831d34482d3cf28fd42080437431677639e0d8` |

## N12 seed measurements

Four MPI ranks ran on node `c128-002`, one per GB200. Local ranks 0–3 selected GPUs 0–3. Global ranks 0–3 used GPUs 0, 2, 1 and 3 respectively. The binary SHA-256 was `82f34391ed2e5a312af143800fd90682bc8ad61a3319763386e5431e22762e3e`. Only the initial-data directory, per-rank block capacity and cycle limit were overridden.

| Measurement | Observed time |
|---|---:|
| Startup through the cycle-0 clock | 258.59 s |
| Cycles 1–130 including outputs and finalization | 397.82 s |
| Sum of the 130 multirate intervals | 47.04 s |
| Cycle 16, the only regrid (754 to 586 blocks) | 158.25 s wall, 157.86 s outside the integrator |
| Cycles 37 and 74, history and horizon output | 1.68 s and 1.50 s outside the integrator |
| Cycle 111, history and shared checkpoint | 87.38 s outside the integrator |
| Cycle 130, final outputs and finalization | 100.40 s outside the integrator |
| Scheduled scri dumps (21), mean preparation plus writing | 0.092 s per dump |

The existing agreement-plan timer recorded three builds on rank 0 during the first regrid, taking 25.51, 23.69 and 13.57 s. After regrid, the solver recorded 12,400 stage evaluations, 1,408 packets, 2,996 pack kernels and 5.27 GB sent per coarse cycle across the four ranks. The cycle-130 checkpoint was preserved off scratch. All 136,575,504 payload doubles were finite. [Checkpoint verification](seed-checkpoint.json)

The 76 cycles from 33 through 127 without scheduled driver output averaged 0.33988 s wall per cycle with a cycle-to-cycle sample SD of 0.01650 s. This is a single-run measurement with both tracker files written on every cycle. The fixed-layout throughput measurement uses three separate restarts from cycle 130 through 245, suppresses tracker writes while preserving their evolution, and discards warmup, finalization and periodic output cycles. Zone-cycles use the active mesh before each step. Speedups and the rank-layout recommendation await those measurements. The supplied 1.31 s/cycle is the Aurora reference.
