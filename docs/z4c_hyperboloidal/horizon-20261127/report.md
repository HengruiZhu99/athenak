# Horizon CUDA baseline, 27 November 2026

The pinned AthenaK revision `c80c114e4c7f219cbbcc209862727fa89d1a3fa2` fails to compile for GB200 with Horizon's nvcc 13.3.73. The campaign stopped before the solver ran. The only gpu-dev job was cancelled while pending, for a total of **0 SU**. [Job ledger](ledger.tsv)

## Baseline build

The source and its pinned Kokkos submodule (`6739bc623081648af9e752b616d9671527922cbf`, version 4.7.2) were exported from Git and verified by archive checksum on Horizon. The Release configuration succeeded with Open MPI 5.0.11, METIS, the Kokkos CUDA and Serial backends, `Kokkos_ARCH_BLACKWELL100=ON`, relaxed constexpr enabled, and host and device floating-point contraction disabled.

While job 109457 waited, a single compiler process built the RHS object on the Horizon login host. It failed in `EvaluateBlocks` at line 66, on the first use of `ind` inside `if constexpr (precomputed_jets)`. The surrounding Kokkos device lambda already names the other captured arguments before that branch. Adding `ind` to the existing `Capture(...)` call is a candidate CUDA compatibility fix. It has not been applied or verified. [Bulk RHS source at the pinned revision](https://github.com/HengruiZhu99/athenak/blob/c80c114e4c7f219cbbcc209862727fa89d1a3fa2/src/z4c_hyperboloidal/z4c_hyperboloidal_rhs.cpp#L43-L67)

```text
z4c_hyperboloidal_rhs.cpp(66): error: An extended __host__ __device__ lambda cannot first-capture variable in constexpr-if context
              const int first = components_per_jet * ind.Slot(f);
                                                     ^
```

## Benchmark status

The N12 evolve-all deck has SHA-256 `cad95d8dac125bffd6ced079dc6ea2c79bff4afc1ec18534aa41ffdab21a0309`. Its declared binary identity matches the existing Horizon export manifest, `7d32c45f889a8c6f224eae2441653dc3cd859e39a904922145968cdbb8c650c2`. The planned smoke layout was one node, four ranks and one GB200 per rank. The available N24 decks on Horizon predate evolve-all. The exact Aurora N24 deck has been requested.

N12 and N24 have no measured Horizon cycle time, zone throughput, baseline profile or GPU identity result at this revision. No optimization was tested, retained or rejected, and there is no measured basis for a rank-layout recommendation. The supplied 1.31 s/cycle is an Aurora N12 reference. No Horizon speedup was estimated.

## Local baseline checks

On macOS arm64 with Kokkos Serial, 27 of 28 registered hyperboloidal and multirate CTest cases passed, including the two- and three-rank migration tests. The unchanged `hyperboloidal_reference_accuracy` case measured a geometry RHS residual of `6.67702e-08` against a `1e-09` limit and a constraint residual of `7.86191e-11` against a `1e-11` limit. The gauge RHS passed. The failure also reproduced in Release builds with and without an explicit `-ffp-contract=off`.

Both radial 1D modes completed 22 steps at `h=0.005`. All evolved fields, rates and scri history entries checked were finite. The optional profile writer explicitly emits NaN for the undefined Hamiltonian and momentum at the outer endpoint. With evolve-all disabled, the scri history and both profiles matched `85e3256fc` byte for byte. Full 3D history and checkpoint identity on Horizon remains untested.
