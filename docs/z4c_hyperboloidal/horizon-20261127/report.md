# Hyperboloidal Z4c performance on Horizon

The CUDA compatibility baseline is `b943db139`. Horizon's nvcc 13.3.73 compiled the corrected bulk RHS object. The full executable and N12 seed are queued on one four-GPU node. The three pending jobs reserve at most 3.67 SU altogether. The account has incurred 0 SU for this campaign as of this update. [Job ledger](ledger.tsv)

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

The N12 seed uses four MPI ranks, one per GB200, through cycle 130 with the original output cadence. Three unprofiled restarts from its final checkpoint will measure through cycle 245. The timing runs suppress the two per-cycle tracker file writes while preserving tracker evolution. The analysis discards warmup, final output and periodic output cycles, and computes zone-cycles from the active mesh before each step. Correctness runs use the original output cadence. The supplied 1.31 s/cycle is the Aurora N12 reference. Horizon timings, speedups and layout comparisons are pending.
